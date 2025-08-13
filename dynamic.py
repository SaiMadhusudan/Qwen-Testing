import os
import sys
import argparse
import subprocess
import json
import tempfile
import shutil
import math
import time
import signal # CHANGED: Added for graceful shutdown handling
import os
from huggingface_hub import constants

# The root directory for the Hugging Face cache
hf_home_path = constants.HF_HOME
print(f"Hugging Face Home (cache root): {hf_home_path}")

# === CONFIGURATION: EDIT THE VALUES BELOW TO MATCH YOUR SETUP ===
CONFIG = {
    # --- Main Paths ---
    "input_folder_path": "./qwen_testing/",
    "output_html_dir": "./outputs",

    # --- Input Data Settings ---
    "type_of_input": "subfolder_images",  # Options: 'images' or 'subfolder_images'
    
    # --- Model and Task Settings ---
    "output_type": "html",  # Options: 'html' or 'doc_type'
    "model_type": "qwen2_5_vl",
    "modality": "image",
    
    # --- Performance/System Settings ---
    "disable_mm_preprocessor_cache": False,
    "min_gpu_memory_mb": 70000, # Min free memory required for a GPU to be used

    # --- Looping Configuration ---
    # CHANGED: This is no longer used by the orchestrator but could be used inside a worker if a delay is desired.
    "LOOP_DELAY_SECONDS": 60, # Time to wait in seconds between processing cycles
}

# === END OF CONFIGURATION ===

# --------------- GPU Selection Utilities (Unchanged) --------------- #
def get_free_gpus(min_mem_mb=8000):
    """
    Returns a list of all free GPU IDs with at least `min_mem_mb` free memory.
    """
    try:
        result = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=index,memory.free',
                '--format=csv,noheader,nounits'
            ],
            capture_output=True, text=True, check=True
        )
        lines = result.stdout.strip().split('\n')
        available = []
        for line in lines:
            if not line: continue
            gpu_id, mem_free = map(str.strip, line.split(','))
            if int(mem_free) > min_mem_mb:
                available.append(gpu_id)
        return available
    except Exception as e:
        print(f"Error while fetching GPU info: {e}")
        return []

def partition_gpus(num_available_gpus):
    """
    Partitions the total number of GPUs into chunks of valid TP sizes.
    e.g., for 7 GPUs, returns [4, 2, 1]
    """
    # Assuming TP sizes must be powers of 2 for this model/setup
    valid_tp_sizes = [16, 8, 4, 2, 1] 
    
    partitions = []
    remaining_gpus = num_available_gpus
    while remaining_gpus > 0:
        size_to_use = 0
        for size in valid_tp_sizes:
            if size <= remaining_gpus:
                size_to_use = size
                break
        
        if size_to_use > 0:
            partitions.append(size_to_use)
            remaining_gpus -= size_to_use
        else:
            # This case should not be reached with the current valid_tp_sizes which includes 1
            print(f"Warning: Could not partition remaining {remaining_gpus} GPUs. Stopping.")
            break
            
    return partitions

# ------------------------- Model Functions (Unchanged) ------------------------- #
def run_qwen2_5_vl(question: str, modality: str, tensor_parallel_size: int, disable_mm_preprocessor_cache: bool):
    from vllm import LLM
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    llm = LLM(
        model=model_name,
        max_model_len=12000,
        max_num_seqs=5,
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
        tensor_parallel_size=tensor_parallel_size,
        disable_mm_preprocessor_cache=disable_mm_preprocessor_cache,
    )
    if modality == "image":
        placeholder = "<|image_pad|>"
    elif modality == "video":
        placeholder = "<|video_pad|>"
    else:
        raise ValueError(f"Unsupported modality: {modality}")
    prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
              f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
              f"{question}<|im_end|>\n"
              "<|im_start|>assistant\n")
    stop_token_ids = None
    return llm, prompt, stop_token_ids

def get_multi_modal_input(args):
    from PIL import Image
    if args.modality == "image":
        if args.output_type == 'html':
            img_question = """
            You are tasked with converting a scanned document image into a fully-renderable HTML document.
            Provide a full HTML representation following detailed instructions.
            """
        elif args.output_type == 'doc_type':
            img_question = """
            Analyze the provided image and classify the document type from a predefined list.
            Provide only the document type in plain text.
            """
        else:
            raise ValueError(f"Unsupported output_type: {args.output_type}")
        return {"question": img_question}
    else:
        raise ValueError(f"Modality {args.modality} is not supported.")

def get_input_filepaths(folder_path, type_of_input):
    filepaths = []
    if type_of_input == 'images':
        for file in os.listdir(folder_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepaths.append(os.path.join(folder_path, file))
    elif type_of_input == 'subfolder_images':
        for imgfolder in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, imgfolder)
            if not os.path.isdir(subfolder_path): continue
            for imgfile in os.listdir(subfolder_path):
                 if imgfile.lower().endswith(('.png', '.jpg', '.jpeg')):
                    filepaths.append(os.path.join(subfolder_path, imgfile))
    return filepaths

# --------------- Worker and Orchestrator Logic --------------- #

# CHANGED: The worker function is heavily modified to run in a continuous loop.
def run_worker(args):
    """
    This function is executed by a worker process. It loads the model ONCE,
    then enters an infinite loop to continuously process the same set of files.
    """
    from PIL import Image
    from vllm import SamplingParams
    Image.MAX_IMAGE_PIXELS = 933120000

    print(f"--- Worker PID {os.getpid()} on GPUs {args.gpus} initializing... ---")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    gpu_ids = args.gpus.split(',')
    tensor_parallel_size = len(gpu_ids)

    with open(args.input_filelist, 'r') as f:
        filepaths_to_process = json.load(f)
    
    if not filepaths_to_process:
        print(f"Worker on GPUs {args.gpus} has no files to process. Exiting.")
        return

    print(f"Worker on GPUs {args.gpus} loaded {len(filepaths_to_process)} images for continuous processing.")

    # --- Load Model and Prepare Prompt ONCE ---
    print(f"Worker on GPUs {args.gpus}: Loading model...")
    try:
        mm_input = get_multi_modal_input(args)
        question = mm_input["question"]
        
        llm, prompt, stop_token_ids = run_qwen2_5_vl(
            question,
            args.modality,
            tensor_parallel_size=tensor_parallel_size,
            disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache
        )
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.9,
            top_k=50,
            max_tokens=4096,
            stop_token_ids=stop_token_ids
        )
        print(f"Worker on GPUs {args.gpus}: Model loaded successfully. Starting infinite processing loop.")
    except Exception as e:
        print(f"FATAL: Worker on GPUs {args.gpus} failed to load model: {e}")
        return # Exit if model loading fails

    # --- Infinite Processing Loop ---
    cycle_count = 0
    while True:
        cycle_start_time = time.time()
        cycle_count += 1
        print(f"\n--- Worker {args.gpus} | Cycle {cycle_count} | Starting at {time.ctime(cycle_start_time)} ---")
        
        inputs = []
        image_names = []  # Keep this for logging purposes

        for filepath in filepaths_to_process:
            try:
                image = Image.open(filepath).convert("RGB")
                inputs.append({
                    "prompt": prompt,
                    "multi_modal_data": {args.modality: image},
                })
                image_names.append(os.path.basename(filepath))
            except Exception as e:
                print(f"Warning (Worker {args.gpus}): Could not load image {filepath}: {e}")
        
        if not inputs:
            print(f"Warning (Worker {args.gpus}): No valid images could be loaded for this cycle. Retrying after a delay.")
            time.sleep(60) # Wait a bit if all images fail to load
            continue

        try:
            # --- RUN INFERENCE ---
            print(f"Worker {args.gpus}: Generating outputs for {len(inputs)} inputs...")
            start_gen_time = time.time()
            outputs = llm.generate(inputs, sampling_params=sampling_params)
            end_gen_time = time.time()
            
            # --- PROCESS OUTPUTS (currently just logging) ---
            # Your request was to not worry about saving, so this part just confirms completion.
            print(f"Worker {args.gpus}: Generation for {len(outputs)} outputs completed in {end_gen_time - start_gen_time:.2f} seconds.")
            
            # You can re-enable saving logic here if you wish.
            # os.makedirs(args.output_html_dir, exist_ok=True)
            # for i, output in enumerate(outputs):
            #     image_name = image_names[i]
            #     generated_text = output.outputs[0].text
            #     output_filename = os.path.splitext(image_name)[0] + '.html'
            #     output_path = os.path.join(args.output_html_dir, output_filename)
            #     with open(output_path, 'w', encoding='utf-8') as f:
            #         f.write(generated_text)
            
        except Exception as e:
            print(f"ERROR: An error occurred during generation for worker on GPUs {args.gpus}: {e}")

        cycle_end_time = time.time()
        print(f"--- Worker {args.gpus} | Cycle {cycle_count} | Finished in {cycle_end_time - cycle_start_time:.2f}s ---")


# CHANGED: The orchestrator now launches persistent workers and waits for a Ctrl+C to terminate them.
def orchestrate_jobs(args):
    """
    Finds free GPUs, partitions them, splits workload, and launches PERSISTENT worker processes.
    It then waits for a KeyboardInterrupt (Ctrl+C) to terminate them.
    """
    print("--- Running in Orchestrator Mode ---")
    
    # Optional: Clean up previous output. You may want to disable this in a continuous run.
    if os.path.exists(args.output_html_dir):
        print(f"Clearing previous outputs from '{args.output_html_dir}'...")
        shutil.rmtree(args.output_html_dir)
    os.makedirs(args.output_html_dir, exist_ok=True)
    
    free_gpus = get_free_gpus(min_mem_mb=args.min_gpu_memory_mb)
    if not free_gpus:
        raise RuntimeError("No suitable free GPUs found.")
        
    num_available = len(free_gpus)
    print(f"Found {num_available} free GPUs: {', '.join(free_gpus)}")

    tp_partitions = partition_gpus(num_available)
    if not tp_partitions:
        raise RuntimeError(f"Could not find a valid way to partition {num_available} GPUs.")
    
    print(f"Partitioning into {len(tp_partitions)} jobs with TP sizes: {tp_partitions}")

    all_filepaths = get_input_filepaths(args.input_folder_path, args.type_of_input)
    if not all_filepaths:
        print("No input files found. Exiting.")
        return
        
    print(f"Found {len(all_filepaths)} total images to process across all workers.")

    gpu_idx_start, file_idx_start, worker_configs = 0, 0, []
    total_gpus_in_partitions = sum(tp_partitions)

    for i, tp_size in enumerate(tp_partitions):
        gpu_idx_end = gpu_idx_start + tp_size
        worker_gpus = free_gpus[gpu_idx_start:gpu_idx_end]
        gpu_idx_start = gpu_idx_end

        if i == len(tp_partitions) - 1:
            worker_files = all_filepaths[file_idx_start:]
        else:
            # Distribute files proportionally to the number of GPUs per worker
            proportion = tp_size / total_gpus_in_partitions
            num_files_for_worker = math.floor(len(all_filepaths) * proportion)
            file_idx_end = file_idx_start + num_files_for_worker
            worker_files = all_filepaths[file_idx_start:file_idx_end]
            file_idx_start = file_idx_end
        
        worker_configs.append({"gpus": ",".join(worker_gpus), "files": worker_files})
    
    processes = []
    temp_dir = tempfile.mkdtemp(prefix="vllm_orch_")
    print(f"Using temporary directory for file lists: {temp_dir}")
    
    try:
        for i, config in enumerate(worker_configs):
            if not config['files']:
                print(f"Skipping worker for GPUs {config['gpus']} as there are no files.")
                continue

            filelist_path = os.path.join(temp_dir, f"worker_{i}_files.json")
            with open(filelist_path, 'w') as f:
                json.dump(config['files'], f)

            cmd = [
                sys.executable, __file__,
                '--worker',
                '--gpus', config['gpus'],
                '--input-filelist', filelist_path,
                '--input-folder-path', args.input_folder_path,
                '--output-html-dir', args.output_html_dir,
                '--type-of-input', args.type_of_input,
                '--output-type', args.output_type,
                '--model-type', args.model_type,
                '--modality', args.modality,
                '--min-gpu-memory-mb', str(args.min_gpu_memory_mb),
            ]
            if args.disable_mm_preprocessor_cache:
                cmd.append('--disable-mm-preprocessor-cache')

            print(f"Launching PERSISTENT worker {i} for TP={len(config['gpus'].split(','))} on GPUs [{config['gpus']}] with {len(config['files'])} files...")
            processes.append(subprocess.Popen(cmd))

        print("\n" + "="*80)
        print("All persistent workers launched. Models are now loading on the GPUs.")
        print("The orchestrator will now wait. Press Ctrl+C to terminate all worker processes.")
        print("="*80 + "\n")
        
        # Wait for all worker processes to terminate. Since they run in an infinite
        # loop, this will wait until they are externally killed (e.g., by Ctrl+C).
        for p in processes:
            p.wait()

    except KeyboardInterrupt:
        print("\nCtrl+C detected by orchestrator. Terminating all worker processes...")
        for p in processes:
            p.terminate() # Send SIGTERM to all child processes
        # Wait for all processes to ensure they have exited
        for p in processes:
            p.wait() 
        print("All worker processes terminated.")
    
    finally:
        print("Orchestrator shutting down. Cleaning up temporary files.")
        shutil.rmtree(temp_dir)

# CHANGED: Main function is simplified to run the orchestration just once.
def main():
    parser = argparse.ArgumentParser(
        description='Orchestrates continuous VLLM inference across all free GPUs. '
                    'Settings are controlled by the CONFIG dictionary at the top of the script.'
    )
    
    parser.add_argument('--input-folder-path', default=CONFIG["input_folder_path"], type=str)
    parser.add_argument('--output-html-dir', default=CONFIG["output_html_dir"], type=str)
    parser.add_argument('--type-of-input', default=CONFIG["type_of_input"], type=str, choices=['images', 'subfolder_images'])
    parser.add_argument('--output-type', default=CONFIG["output_type"], type=str, choices=['html', 'doc_type'])
    parser.add_argument('--model-type', default=CONFIG["model_type"], type=str)
    parser.add_argument('--modality', default=CONFIG["modality"], type=str)
    parser.add_argument('--disable-mm-preprocessor-cache', action='store_true', default=CONFIG["disable_mm_preprocessor_cache"])
    parser.add_argument('--min-gpu-memory-mb', default=CONFIG["min_gpu_memory_mb"], type=int)
    parser.add_argument('--loop-delay-seconds', default=CONFIG["LOOP_DELAY_SECONDS"], type=int)


    # Internal arguments for worker processes
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--gpus', type=str, help=argparse.SUPPRESS)
    parser.add_argument('--input-filelist', type=str, help=argparse.SUPPRESS)

    args = parser.parse_args()

    if CONFIG["disable_mm_preprocessor_cache"]:
        args.disable_mm_preprocessor_cache = True

    if args.worker:
        # Code path for the subprocesses launched by the orchestrator
        run_worker(args)
    else:
        # Code path for the main script: launch the orchestrator once
        try:
            print("="*80)
            print("Starting VLLM Continuous Processing Orchestrator")
            print(f"Start Time: {time.ctime()}")
            print("="*80)
            
            orchestrate_jobs(args)
            
            print("\n" + "="*80)
            print("Orchestration complete. All worker processes have been terminated.")
            print("="*80)

        except Exception as e:
            print("\n" + "!"*80)
            print(f"A fatal error occurred during orchestration setup: {e}")
            print("Exiting.")
            print("!"*80)

if __name__ == "__main__":
    main()
