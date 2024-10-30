import os
import argparse
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch

video = None
def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    from main import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_custom_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_custom_nodes()


from nodes import (
    CLIPTextEncode,
    EmptyLatentImage,
    VAELoader,
    CLIPSetLastLayer,
    LoadImage,
    CLIPVisionLoader,
    ControlNetApplyAdvanced,
    VAEEncode,
    VAEEncodeForInpaint,
    KSampler,
    NODE_CLASS_MAPPINGS,
    CheckpointLoaderSimple,
    ConditioningCombine,
    VAEDecode,
)


def main():
    print(args.video)
    import_custom_nodes()
    with torch.inference_mode():
        loadimage = LoadImage()
        loadimage_6 = loadimage.load_image(image="douki (2).png")

        ttn_int = NODE_CLASS_MAPPINGS["ttN int"]()
        ttn_int_451 = ttn_int.convert(int=44)

        ttn_int_456 = ttn_int.convert(int=12)

        checkpointloadersimple = CheckpointLoaderSimple()
        checkpointloadersimple_593 = checkpointloadersimple.load_checkpoint(
            ckpt_name="mistoonAnime_v20.safetensors"
        )

        clipsetlastlayer = CLIPSetLastLayer()
        clipsetlastlayer_35 = clipsetlastlayer.set_last_layer(
            stop_at_clip_layer=-2,
            clip=get_value_at_index(checkpointloadersimple_593, 1),
        )

        cliptextencode = CLIPTextEncode()
        cliptextencode_491 = cliptextencode.encode(
            text="office lady, 1girl, smiling, portrait, headshot, office, blue shirt, smoking",
            clip=get_value_at_index(clipsetlastlayer_35, 0),
        )

        cliptextencode_492 = cliptextencode.encode(
            text="easynegative, ball bearings, balls, hair pins, bad hands, worst quality, low quality , medium quality , high quality , lowres , bad anatomy, blurry, artifacts, pig tails, (blue hair)",
            clip=get_value_at_index(clipsetlastlayer_35, 0),
        )

        controlnetloaderadvanced = NODE_CLASS_MAPPINGS["ControlNetLoaderAdvanced"]()
        controlnetloaderadvanced_516 = controlnetloaderadvanced.load_controlnet(
            control_net_name="control_v11p_sd15s2_lineart_anime_fp16.safetensors"
        )

        vhs_loadvideo = NODE_CLASS_MAPPINGS["VHS_LoadVideo"]()
        vhs_loadvideo_13 = vhs_loadvideo.load_video(video=video, 
        force_size="Custom",
        frame_load_cap =44,
        force_rate= 12,
        select_every_nth= 1,
        custom_width = 512,
        custom_height = 512,
        skip_first_frames = 1

        )

        cr_image_input_switch_4_way = NODE_CLASS_MAPPINGS[
            "CR Image Input Switch (4 way)"
        ]()
        cr_image_input_switch_4_way_476 = cr_image_input_switch_4_way.switch(
            Input=1, image1=get_value_at_index(vhs_loadvideo_13, 0)
        )

        vaeloader = VAELoader()
        vaeloader_669 = vaeloader.load_vae(vae_name="orangemix.vae.pt")

        vaeencode = VAEEncode()
        vaeencode_532 = vaeencode.encode(
            pixels=get_value_at_index(cr_image_input_switch_4_way_476, 0),
            vae=get_value_at_index(vaeloader_669, 0),
        )

        clipvisionloader = CLIPVisionLoader()
        clipvisionloader_586 = clipvisionloader.load_clip(
            clip_name="SD1.5/pytorch_model.bin"
        )

        ipadaptermodelloader = NODE_CLASS_MAPPINGS["IPAdapterModelLoader"]()
        ipadaptermodelloader_590 = ipadaptermodelloader.load_ipadapter_model(
            ipadapter_file="ip-adapter_sd15.bin"
        )

        # monochromaticclip = NODE_CLASS_MAPPINGS["MonochromaticClip"]()
        # monochromaticclip_638 = monochromaticclip.monochromatic_clip(
        #     channel="greyscale", threshold=0
        # )

        midas_depthmappreprocessor = NODE_CLASS_MAPPINGS["MiDaS-DepthMapPreprocessor"]()
        midas_depthmappreprocessor_644 = midas_depthmappreprocessor.execute(
            a=6.283185307179586,
            bg_threshold=0.2,
            resolution=512,
            image=get_value_at_index(cr_image_input_switch_4_way_476, 0),
        )

        image_filters = NODE_CLASS_MAPPINGS["Image Filters"]()
        image_filters_650 = image_filters.image_filters(
            brightness=-0.3,
            contrast=1.24,
            saturation=1,
            sharpness=1,
            blur=0,
            gaussian_blur=0,
            edge_enhance=0,
            image=get_value_at_index(midas_depthmappreprocessor_644, 0),
        )

        monochrombatch = NODE_CLASS_MAPPINGS["MonoChromBatch"]()
        monochrombatch_680 = monochrombatch.MonoChrom_Batch(
            channel="greyscale",
            threshold=0,
            image=get_value_at_index(image_filters_650, 0),
        )

        imagetomask = NODE_CLASS_MAPPINGS["ImageToMask"]()
        imagetomask_639 = imagetomask.image_to_mask(
            channel="blue", image=get_value_at_index(monochrombatch_680, 0)
        )

        vaeencodeforinpaint = VAEEncodeForInpaint()
        vaeencodeforinpaint_651 = vaeencodeforinpaint.encode(
            grow_mask_by=12,
            pixels=get_value_at_index(image_filters_650, 0),
            vae=get_value_at_index(vaeloader_669, 0),
            mask=get_value_at_index(imagetomask_639, 0),
        )

        emptylatentimage = EmptyLatentImage()
        emptylatentimage_663 = emptylatentimage.generate(
            width=512, height=512, batch_size=1
        )

        loadimage_668 = loadimage.load_image(image="target-4 (1) (1).png")

        ade_animatediffuniformcontextoptions = NODE_CLASS_MAPPINGS[
            "ADE_AnimateDiffUniformContextOptions"
        ]()
        ade_animatediffuniformcontextoptions_677 = (
            ade_animatediffuniformcontextoptions.create_options(
                context_schedule="uniform", fuse_method="flat", context_length=24, context_stride=1, context_overlap=6,closed_loop=False
            )
        )

        prepimageforclipvision = NODE_CLASS_MAPPINGS["PrepImageForClipVision"]()
        ade_animatediffloadergen1 = NODE_CLASS_MAPPINGS["ADE_AnimateDiffLoaderGen1"]()
        ipadapterapply = NODE_CLASS_MAPPINGS["IPAdapterApply"]()
        freeu_v2 = NODE_CLASS_MAPPINGS["FreeU_V2"]()
        modelsamplingdiscrete = NODE_CLASS_MAPPINGS["ModelSamplingDiscrete"]()
        batchpromptschedule = NODE_CLASS_MAPPINGS["BatchPromptSchedule"]()
        conditioningcombine = ConditioningCombine()
        cr_conditioning_input_switch = NODE_CLASS_MAPPINGS[
            "CR Conditioning Input Switch"
        ]()
        lineartpreprocessor = NODE_CLASS_MAPPINGS["LineArtPreprocessor"]()
        controlnetapplyadvanced = ControlNetApplyAdvanced()
        ksampler = KSampler()
        vaedecode = VAEDecode()
        vhs_videocombine = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()
        masktoimage = NODE_CLASS_MAPPINGS["MaskToImage"]()

        #for q in range(10):
        prepimageforclipvision_608 = prepimageforclipvision.prep_image(
            interpolation="NEAREST",
            crop_position="top",
            sharpening=0.16,
            image=get_value_at_index(loadimage_6, 0),
        )

        ade_animatediffloadergen1_671 = (
            ade_animatediffloadergen1.load_mm_and_inject_params(
                model_name="mm_sd_v15_v2.ckpt",
                beta_schedule="sqrt_linear (AnimateDiff)",
                model=get_value_at_index(checkpointloadersimple_593, 0),
                context_options=get_value_at_index(
                    ade_animatediffuniformcontextoptions_677, 0
                ),
            )
        )

        ipadapterapply_585 = ipadapterapply.apply_ipadapter(
            weight=0.71,
            noise=0.4,
            weight_type="original",
            start_at=0,
            end_at=1,
            unfold_batch=False,
            ipadapter=get_value_at_index(ipadaptermodelloader_590, 0),
            clip_vision=get_value_at_index(clipvisionloader_586, 0),
            image=get_value_at_index(prepimageforclipvision_608, 0),
            model=get_value_at_index(ade_animatediffloadergen1_671, 0),
        )

        freeu_v2_632 = freeu_v2.patch(
            b1=1.22,
            b2=1.32,
            s1=0.91,
            s2=0.19,
            model=get_value_at_index(ipadapterapply_585, 0),
        )

        modelsamplingdiscrete_554 = modelsamplingdiscrete.patch(
            sampling="lcm", zsnr=False, model=get_value_at_index(freeu_v2_632, 0)
        )

        batchpromptschedule_195 = batchpromptschedule.animate(
            text = '"0" :"face, looking at viewer, open_eyes, smile","12" :"face,looking at viewer, open_eyes, smile, happy","24" :"face,looking at viewer, closed_eyes, laughing","36" :"head tilt left, face,looking at viewer, open_eyes, smile, happy","48" :"head center ,looking at viewer, closed_eyes, laughing","60" :"head center, face,looking at viewer, open_eyes, smile"',
            max_frames=72,
            print_output=False,
            pre_text="",
            app_text="0",
            start_frame=0,
            pw_a=0,
            pw_b=0,
            pw_c=0,
            pw_d=0,
            clip=get_value_at_index(clipsetlastlayer_35, 0),
        )

        conditioningcombine_213 = conditioningcombine.combine(
            conditioning_1=get_value_at_index(cliptextencode_491, 0),
            conditioning_2=get_value_at_index(batchpromptschedule_195, 0),
        )

        cr_conditioning_input_switch_211 = cr_conditioning_input_switch.switch(
            Input=1,
            conditioning1=get_value_at_index(conditioningcombine_213, 0),
            conditioning2=get_value_at_index(cliptextencode_491, 0),
        )

        lineartpreprocessor_534 = lineartpreprocessor.execute(
            resolution=512,
            image=get_value_at_index(cr_image_input_switch_4_way_476, 0),
            coarse = "disable"
        )

        controlnetapplyadvanced_509 = controlnetapplyadvanced.apply_controlnet(
            strength=0.7000000000000001,
            start_percent=0,
            end_percent=1,
            positive=get_value_at_index(cr_conditioning_input_switch_211, 0),
            negative=get_value_at_index(cliptextencode_492, 0),
            control_net=get_value_at_index(controlnetloaderadvanced_516, 0),
            image=get_value_at_index(lineartpreprocessor_534, 0),
        )

        ksampler_623 = ksampler.sample(
            seed=random.randint(1, 2**64),
            steps=10,
            cfg=10,
            sampler_name="lcm",
            scheduler="exponential",
            denoise=1,
            model=get_value_at_index(modelsamplingdiscrete_554, 0),
            positive=get_value_at_index(controlnetapplyadvanced_509, 0),
            negative=get_value_at_index(controlnetapplyadvanced_509, 1),
            latent_image=get_value_at_index(vaeencodeforinpaint_651, 0),
        )

        vaedecode_502 = vaedecode.decode(
            samples=get_value_at_index(ksampler_623, 0),
            vae=get_value_at_index(vaeloader_669, 0),
        )

        vhs_videocombine_63 = vhs_videocombine.combine_video(
            frame_rate=get_value_at_index(ttn_int_456, 0),
            loop_count=0,
            filename_prefix="IF_animator",
            format="video/webm",
            pingpong=False,
            save_output=True,
            images=get_value_at_index(vaedecode_502, 0),
            unique_id=6514718207850174548,
        )

        vhs_videocombine_136 = vhs_videocombine.combine_video(
            frame_rate=get_value_at_index(ttn_int_456, 0),
            loop_count=0,
            filename_prefix="IF_OP",
            format="image/gif",
            pingpong=False,
            save_output=False,
            images=get_value_at_index(lineartpreprocessor_534, 0),
            unique_id=16370497527926580347,
        )

        vhs_videocombine_368 = vhs_videocombine.combine_video(
            frame_rate=get_value_at_index(ttn_int_456, 0),
            loop_count=0,
            filename_prefix="AnimateDiff",
            format="image/gif",
            pingpong=False,
            save_output=True,
            images=get_value_at_index(vaedecode_502, 0),
            unique_id=2646940368614063445,
        )

        emptylatentimage_592 = emptylatentimage.generate(
            width=512, height=512, batch_size=get_value_at_index(ttn_int_451, 0)
        )

        masktoimage_683 = masktoimage.mask_to_image(
            mask=get_value_at_index(imagetomask_639, 0)
        )


if __name__ == "__main__":
    main()
