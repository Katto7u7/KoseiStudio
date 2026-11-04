import re


def _find(pattern, text, default=""):
    m = re.search(pattern, text)
    return m.group(1) if m else default


def extract_data_from_api_response(api_response):
    match = re.search(r"info=\{.*?\}", api_response, re.DOTALL)
    if not match:
        print("No se encontró la cadena 'info=' en el formato esperado.")
        return None

    info_str = match.group(0)[5:-1]

    try:
        prompt = _find(r"'prompt': '(.*?)'", info_str)
        all_prompts = _find(r"'all_prompts': \[(.*?)\]", info_str)
        negative_prompt = _find(r"'negative_prompt': '(.*?)'", info_str)
        all_negative_prompts = _find(r"'all_negative_prompts': \[(.*?)\]", info_str)
        seed = _find(r"'seed': (\d+)", info_str)
        all_seeds = _find(r"'all_seeds': \[(.*?)\]", info_str)
        subseed = _find(r"'subseed': (\d+)", info_str)
        all_subseeds = _find(r"'all_subseeds': \[(.*?)\]", info_str)
        subseed_strength = _find(r"'subseed_strength': (.*?),", info_str)
        width = _find(r"'width': (\d+),", info_str)
        height = _find(r"'height': (\d+),", info_str)
        sampler_name = _find(r"'sampler_name': '(.*?)'", info_str)
        cfg_scale = _find(r"'cfg_scale': (.*?),", info_str)
        steps = _find(r"'steps': (\d+),", info_str)
        batch_size = _find(r"'batch_size': (\d+),", info_str)
        restore_faces = _find(r"'restore_faces': (.*?),", info_str)
        face_restoration_model = _find(r"'face_restoration_model': (.*?),", info_str)
        sd_model_name = _find(r"'sd_model_name': '(.*?)'", info_str)
        sd_model_hash = _find(r"'sd_model_hash': '(.*?)'", info_str)
        sd_vae_name = _find(r"'sd_vae_name': (.*?),", info_str)
        sd_vae_hash = _find(r"'sd_vae_hash': (.*?),", info_str)
        seed_resize_from_w = _find(r"'seed_resize_from_w': (\d+),", info_str)
        seed_resize_from_h = _find(r"'seed_resize_from_h': (\d+),", info_str)
        denoising_strength = _find(r"'denoising_strength': (.*?),", info_str)

        extra_match = re.search(r"'extra_generation_params': \{(.*?)\}", info_str)
        extra_generation_params = extra_match.group(1) if extra_match else "{}"

        return [
            f"prompt: '{prompt}',",
            f"all_prompts: {all_prompts},",
            f"negative_prompt: '{negative_prompt}',",
            f"all_negative_prompts: {all_negative_prompts},",
            f"seed: {seed},",
            f"all_seeds: {all_seeds},",
            f"subseed: {subseed},",
            f"all_subseeds: {all_subseeds},",
            f"subseed_strength: {subseed_strength},",
            f"width: {width},",
            f"height: {height},",
            f"sampler_name: '{sampler_name}',",
            f"cfg_scale: {cfg_scale},",
            f"steps: {steps},",
            f"batch_size: {batch_size},",
            f"restore_faces: {restore_faces},",
            f"face_restoration_model: {face_restoration_model},",
            f"sd_model_name: '{sd_model_name}',",
            f"sd_model_hash: '{sd_model_hash}',",
            f"sd_vae_name: {sd_vae_name},",
            f"sd_vae_hash: {sd_vae_hash},",
            f"seed_resize_from_w: {seed_resize_from_w},",
            f"seed_resize_from_h: {seed_resize_from_h},",
            f"denoising_strength: {denoising_strength},",
            f"extra_generation_params: {extra_generation_params}",
        ]
    except Exception as exc:
        print(f"[extraerPrompts] parse error: {exc}")
        return None


def write_data_to_txt(data_list, filename):
    with open(filename, "w") as file:
        file.write("[\n")
        for item in data_list:
            file.write(f"    {item}\n")
        file.write("]")
