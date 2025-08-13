
import os
import sys
import argparse  # for early parsing
import subprocess

# --------------- GPU Selection Utilities --------------- #
def get_free_gpus(required_num=1, min_mem_mb=8000):
    """Returns a comma-separated string of free GPU IDs with at least `min_mem_mb` memory."""
    try:
        result = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=index,memory.free',
                '--format=csv,noheader,nounits'
            ],
            capture_output=True, text=True, check=True
        )
        # Each line: "0, 82373"
        lines = result.stdout.strip().split('\n')
        available = []
        for line in lines:
            gpu_id, mem_free = map(str.strip, line.split(','))
            if int(mem_free) > min_mem_mb:
                available.append(gpu_id)
            if len(available) == required_num:
                break
        return ','.join(available) if len(available) == required_num else None
    except Exception as e:
        print(f"Error while fetching GPU info: {e}")
        return None

# --------------- Parse command line for GPU assignment (before imports) --------------- #
basic_parser = argparse.ArgumentParser(add_help=False)
basic_parser.add_argument('--gpus', type=str, default=None)
args, _ = basic_parser.parse_known_args()

if args.gpus:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    print("Using user-specified GPUs:", args.gpus)
else:
    # Adjust these as per resource needs (Qwen 7B typically: required_num=4, min_mem_mb=70000)
    free_gpus = get_free_gpus(required_num=4, min_mem_mb=70000)
    if free_gpus is None:
        raise RuntimeError("No suitable free GPUs found.")
    os.environ["CUDA_VISIBLE_DEVICES"] = free_gpus
    print("Automatically selected free GPUs:", free_gpus)
# Now, the CUDA environment is set up before the next heavy imports.

# ------------------------- Rest of Imports ------------------------- #
import random
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from PIL import Image
Image.MAX_IMAGE_PIXELS = 933120000
from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset
from vllm.utils import FlexibleArgumentParser
import time

# ------------------------- Model Functions ------------------------- #

def run_qwen2_5_vl(question: str, modality: str, tensor_parallel_size: int, disable_mm_preprocessor_cache: bool):
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
    if args.modality == "image":
        image = Image.open("./qwen_testing/22-04-2016-p-/22-04-2016-p-_page_1.png").convert("RGB")
        if args.output_type == 'html':
            img_question = """
            ... (omitted for brevity: keep your HTML instructions here)
            """
        elif args.output_type == 'doc_type':
            img_question = """
            ... (omitted for brevity: keep your doc_type instructions here)
            """
        return {
            "data": image,
            "question": img_question,
        }
    if args.modality == "video":
        video = VideoAsset(name="sample_demo_1.mp4",
                           num_frames=args.num_frames).np_ndarrays
        vid_question = "Why is this video funny?"
        return {
            "data": video,
            "question": vid_question,
        }
    raise ValueError(f"Modality {args.modality} is not supported.")

def apply_image_repeat(folder_path, prompt, modality):
    inputs = []
    for file in os.listdir(folder_path):
        if file.endswith(('.png','.jpg','.jpeg')): 
            try:      
                image=Image.open(os.path.join(folder_path,file))
                inputs.append({
                    "prompt": prompt,
                    "multi_modal_data": {
                        modality: image
                    },
                    "image_name":file
                })
            except Exception as e:
                print(f"error loading{file}:{e}")
    return inputs    

def folders_pdf(folder_path, prompt, modality):
    inputs = []
    for imgfolder in os.listdir(folder_path):
        for imgfile in os.listdir(os.path.join(folder_path,imgfolder)):
            name= os.path.join(folder_path,imgfolder,imgfile)
            if name.endswith(('.png','.jpg','.jpeg')):
                image=Image.open(os.path.join(folder_path,imgfolder,imgfile))
                inputs.append({
                    "prompt": prompt,
                    "multi_modal_data": {
                        modality: image
                    },
                    "image_name":imgfile
                })
    return inputs               

model_example_map = {
    "qwen2_5_vl": run_qwen2_5_vl,
}

# -------------------- MAIN: Parse all other args and run -------------------- #
def main():
    parser = FlexibleArgumentParser(
        description='Demo on using vLLM for offline inference with vision language models for text generation')
    parser.add_argument('--gpus', type=str, default=None,
        help='Comma-separated GPU IDs to use (e.g., "0,1,2"). Leave unset for auto-selection.')
    parser.add_argument('--model-type', '-m', type=str, default="qwen2_5_vl",
                        choices=model_example_map.keys(),
                        help='Huggingface "model_type".')
    parser.add_argument('--num-prompts', type=int, default=2,
                        help='Number of prompts to run.')
    parser.add_argument('--modality', type=str, default="image",
                        choices=['image', 'video'],
                        help='Modality of the input.')
    parser.add_argument('--num-frames', type=int, default=16,
                        help='Number of frames to extract from the video.')
    parser.add_argument('--image-repeat-prob', type=float, default=None,
        help='Simulates hit-ratio for multi-modal preprocessor cache if enabled')
    parser.add_argument('--disable-mm-preprocessor-cache', action='store_true',
        help='Disable multi-modal preprocessor cache')
    parser.add_argument('--time-generate', action='store_true',
        help='If True, print the total generate() call time')
    parser.add_argument('--type-of-input', type = str, default = 'subfolder_images',
        choices=['images','subfolder_images'])
    parser.add_argument('--input-folder-path', default="./qwen_testing/", type=str)
    parser.add_argument('--output-html-dir', default="./outputs", type=str)
    parser.add_argument('--output-type', type=str, default="html", choices=['html','doc_type'])
    args = parser.parse_args()

    # NOW count visible GPUs and use for tensor_parallel_size
    selected_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(',')
    tensor_parallel_size = len(selected_gpus)

    # Main body
    model = args.model_type
    if model not in model_example_map:
        raise ValueError(f"Model type {model} is not supported.")
    modality = args.modality
    type_of_input = args.type_of_input
    input_folder_path = args.input_folder_path
    output_html_dir = args.output_html_dir
    output_type = args.output_type
    os.makedirs(output_html_dir, exist_ok=True)
    mm_input = get_multi_modal_input(args)
    data = mm_input["data"]
    question = mm_input["question"]
    llm, prompt, stop_token_ids = model_example_map[model](
        question, modality, tensor_parallel_size=tensor_parallel_size, 
        disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache)

    sampling_params = SamplingParams(
        temperature=0.8,        
        top_p=0.9,              
        top_k=50,               
        max_tokens=20,         
        stop_token_ids=stop_token_ids  
    )
    assert args.num_prompts > 0
    if type_of_input == 'images':
        inputs = apply_image_repeat(input_folder_path,prompt,modality)
    elif type_of_input == 'subfolder_images':
        inputs = folders_pdf(input_folder_path,prompt,modality) 
    start_time = time.time()
    while True:
        outputs = llm.generate(inputs, sampling_params=sampling_params)
        # (You should implement result saving/break/outputs as desired.)

if __name__ == "__main__":
    main()
