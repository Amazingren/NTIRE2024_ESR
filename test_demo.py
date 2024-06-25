import os.path
import logging
import torch
import argparse
import json
import glob

from pprint import pprint
from fvcore.nn import FlopCountAnalysis
from utils.model_summary import get_model_activation, get_model_flops
from utils import utils_logger
from utils import utils_image as util

from torch.nn import functional as F # For team34

def select_model(args, device):
    # Model ID is assigned according to the order of the submissions.
    # Different networks are trained with input range of either [0,1] or [0,255]. The range is determined manually.
    model_id = args.model_id
    if model_id == 0:
        # Baseline: Winner of the NTIRE 2022 Efficient SR Challenge 
        # RLFN: Residual Local Feature Network for Efficient Super-Resolution
        # arXiv: https://arxiv.org/pdf/2205.07514.pdf
        # Original Code: https://github.com/bytedance/RLFN
        # Ckpts: rlfn_ntire_x4.pth
        from models.team00_RLFN import RLFN_Prune
        name, data_range = f"{model_id:02}_RLFN_baseline", 255.0
        model_path = os.path.join('model_zoo', 'team00_rlfn.pth')
        model = RLFN_Prune()
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 1:
        from models.team01_PFDNLITE import PFDN_Lite
        name, data_range = f"{model_id:02}_PFDN_Lite", 1.0
        model_path = os.path.join('model_zoo', 'team01_pfdnlite.pth')
        model = PFDN_Lite()
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 3:
        from models.team03_LKFN import LKFN
        name, data_range = f"{model_id:02}_LKFN", 1.0
        model_path = os.path.join("model_zoo", "team03_lkfn.pth")
        model = LKFN()
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 4:
        from models.team04_RCN import RCN
        name, data_range = f"{model_id:02}_RCN", 1.0
        model_path = os.path.join('model_zoo', 'team04_rcn.pth')
        model = RCN(feature_channels = 40, mid_channels = 40)
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 5:
        from models.team05_RLFF import RLFN_mogai
        name, data_range = f"{model_id:02}_RLFN_mogai", 1.0
        model_path = os.path.join('model_zoo', 'team05_RLFN_mogai.pth')
        model = RLFN_mogai()
        model.load_state_dict(torch.load(model_path)['params'], strict=True)
    elif model_id == 7:
        from models.team07_DVMSR import DVMSR
        name, data_range = f"{model_id:02}_DVMSR", 1.0
        model_path = os.path.join('model_zoo', 'team07_DVMSR.pth')
        model = DVMSR()
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 10:
        from models.team10_SMRN import SMRN
        name, data_range = f"{model_id:02}_SMRN", 1.0
        model_path = os.path.join('model_zoo', 'team10_SMRN.pth')
        model = SMRN()
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 11:
        from models.team11_EFQN import QuickSRNetLarge
        name, data_range = f"{model_id:02}_EFQN", 1.0
        model_path = os.path.join('model_zoo', 'team11_EFQN.pt')
        model = QuickSRNetLarge()
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 13:
        from models.team13_RLFN_B import RLFN_B
        # from models.team00_RLFN import RLFN_Prune
        name, data_range = f"{model_id:02}_RLFN_B", 255.0
        # model_path1 = os.path.join('model_zoo', 'team13_model1.pth')
        # model_path2 = os.path.join('model_zoo', 'team13_model2.pth')
        # model1 = RLFN_Prune()
        # model1.load_state_dict(torch.load(model_path1), strict=True)
        # model2 = RLFN_B()
        # model2.load_state_dict(torch.load(model_path2), strict=True)
        model_path = os.path.join('model_zoo', 'team13_model2.pth')
        model = RLFN_B()
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 15:
        from models.team15_EMaxGMan import EMaxGMan
        name, data_range = f"{model_id:02}_EMaxGMan", 1.0
        model_path = os.path.join('model_zoo', 'team15_EMaxGMan.pth')
        model = EMaxGMan()
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 16:
        from models.team16_LightRLFN import LightRLFN
        name, data_range = f"{model_id:02}_LightRLFN", 1.0
        model_path = os.path.join('model_zoo', 'team16_LightRLFN.pth')
        model = LightRLFN()
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 17:
        from models.team17_AdvancedSR import ASR_Prune
        name, data_range = f"{model_id:02}_ASR_Prune", 255.0
        model_path = os.path.join('model_zoo', 'team17_AdvancedSR.pth')
        model = ASR_Prune()
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 18:
        from models.team18_MagicSR import MagicSR
        name, data_range = f"{model_id:02}_MagicSR", 1.0
        model_path = os.path.join('model_zoo', 'team18_magicsr.pth')
        model = MagicSR(training_img_size= 64,
                       ngrams=(2,2,2,2),
                       in_chans= 3,
                       embed_dim= 64,
                       depths=(6,4,4),
                       num_heads=(6,4,4),
                       head_dim= None,
                       dec_dim= 64,
                       dec_depths= 6,
                       dec_num_heads= 6,
                       dec_head_dim= None,
                       target_mode= 'light_x4',
                       window_size= 8,
                       mlp_ratio= 2.0,
                       qkv_bias= True,
                       img_norm= True,
                       drop_rate= 0.0,
                       attn_drop_rate= 0.0,
                       drop_path_rate= 0.0)
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 19:
        from models.team19_DIRN import DIRN
        name, data_range = f"{model_id:02}_DIRN", 1.0
        model_path = os.path.join('model_zoo', 'team19_DIRN.pth')
        model = DIRN()
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 21:
        from models.team21_SlimRLFN import SlimRLFN
        name, data_range = f"{model_id:02}_SlimRLFN", 255.0
        # Prune
        # model_path = os.path.join('model_zoo', 'team21_slimrlfn_prune.pth')
        # model = SlimRLFN(prune_channels=[30 - v for v in [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0]])
        # Normal
        model_path = os.path.join('model_zoo', 'team21_slimrlfn.pth')
        model = SlimRLFN()
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 22:
        from models.team22_GFMN import GFMN
        name,data_range=f"{model_id:02}_GFMN", 1.0
        model_path = os.path.join('model_zoo','team22_gfmn.pth')
        model = GFMN()
        model.load_state_dict(torch.load(model_path)['params'], strict=True)
    elif model_id == 23:
        from models.team23_safmnpp import SAFMNPP
        name, data_range = f"{model_id:02}_SAFMNPP", 1.0
        model_path = os.path.join('model_zoo', 'team23_safmnpp.pth')
        model = SAFMNPP(dim=36, n_blocks=6, ffn_scale=1.5, upscaling_factor=4)
        model.load_state_dict(torch.load(model_path)['params_ema'], strict=True)
    elif model_id == 24:
        from models.team24_smfan import SMFAN
        name, data_range = f"{model_id:02}_SMFA", 1.0
        model_path = os.path.join('model_zoo', 'team24_smfan.pth')
        model = SMFAN()
        model.load_state_dict(torch.load(model_path)['params_ema'], strict=True)
    elif model_id == 25:
        from models.team25_EERN import EERN
        name, data_range = f"{model_id:02}_EERN", 255.0
        model_path = os.path.join('model_zoo', 'team25_EERN.pt')
        model = EERN()
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 26:
        from models.team26_RDEN import RDEN
        name, data_range = f"{model_id:02}_RDEN", 1.0
        model_path = os.path.join('model_zoo', 'team26_RDEN.pth')
        model = RDEN()
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 28:
        from models.team28_MSN import MSN
        name, data_range = f"{model_id:02}_MSN", 1.0
        model_path = os.path.join('model_zoo', 'team28_MSN.pth')
        model = MSN()
        model.load_state_dict(torch.load(model_path)['params_ema'], strict=True)
    elif model_id == 29:
        from models.team29_R2Net import R2Net
        name, data_range = f"{model_id:02}_R2Net", 1.0
        model = R2Net(in_channels=3,out_channels=3,
                    feature_channels=48,upscale=4,bias=False,rep='plain')
        model_path = os.path.join('model_zoo', 'team29_R2Net.pth')#os.path.join('model_zoo', 'R2Net_for_submit.pth')
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 30: 
        from models.team30_RFDN import RFDN
        name, data_range = f"{model_id:02}_RFDN", 255.0 # Maybe it can be 1.0
        model = RFDN()
        model_path = os.path.join('model_zoo', 'team30_RFDN.pkl')#os.path.join('model_zoo', 'R2Net_for_submit.pth')
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 31:
        from models.team31_AGDN import AGDN
        name, data_range = f"{model_id:02}_AGDN", 1.0
        model_path = os.path.join('model_zoo', 'team31_agdn.pth')
        model = AGDN(num_feat=24, upscale=4)
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 33:
        from models.team33_IFADNet import IFADNet
        name, data_range = f"{model_id:02}_IFADNet", 1.0
        model = IFADNet(deploy=True)
        model_path = os.path.join('model_zoo', 'team33_IFADNet.pth')
        model.load_state_dict(torch.load(model_path)["params_ema"], strict=True)
    elif model_id == 34:
        from models.team34_craft import CRAFT
        name, data_range = f"{model_id:02}_craft", 255.0
        model_path = os.path.join('model_zoo', 'team34_craft.pth')
        model = CRAFT(upscale=4,
            in_chans=3,
            img_size=64,
            window_size=16,
            img_range=1.,
            depths=[2, 2, 2, 2],
            embed_dim=48,
            num_heads=[6, 6, 6, 6],
            mlp_ratio=2,
            resi_connection='1conv')
        loadnet = torch.load(model_path)
        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        else:
            keyname = 'params'
        model.load_state_dict(loadnet[keyname], strict=True)
    elif model_id == 38:
        import importlib
        model_module = importlib.import_module(f'models.team{model_id:02}_SPAN')
        name, data_range = f"{model_id:02}_SPAN", 1.0
        model = getattr(model_module, f'SPAN30')(3, 3, upscale=4, feature_channels=28).eval().to(device)
        model_path = os.path.join('model_zoo', f'team38_span_ch28_slim.pth')
        stat_dict = torch.load(model_path)
        model.load_state_dict(stat_dict, strict=False)
    elif model_id == 39:
        import importlib
        model_module = importlib.import_module(f'models.team{model_id:02}_SPANtiny')
        name, data_range = f"{model_id:02}_SPANtiny", 1.0
        model = getattr(model_module, f'SPAN30')(3, 3, upscale=4, feature_channels=26).eval().to(device)
        model_path = os.path.join('model_zoo', f'team39_spantiny_ch26_slim.pth')
        stat_dict = torch.load(model_path)
        model.load_state_dict(stat_dict, strict=False)
    elif model_id == 41:
        from models.team41_HARN import HARN
        name, data_range = f"{model_id:01}_HARN", 1.0
        model_path = os.path.join('model_zoo', 'team41_harn.pth')
        model = HARN()
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 42:
        from models.team42_DepthRLFN import RLFN_PruneLocal
        name, data_range = f"{model_id:02}_DepthRLFN", 1.0
        model_path = os.path.join('model_zoo', 'team42_DepthRLFN.pth')
        model = RLFN_PruneLocal()
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 43:
        from models.team43_EGFMN import EGFMN
        name, data_range = f"{model_id:02}_EGFMN", 1.0
        model_path = os.path.join('model_zoo', 'team43_EGFMN.pth')
        model = EGFMN()
        model.load_state_dict(torch.load(model_path)['params_ema'], strict=True)
    elif model_id == 44:
        from models.team44_LSANet import LSANet
        name, data_range = f"{model_id:02}_LSANet", 1.0
        model = LSANet(deploy=True)
        model_path = os.path.join('model_zoo', 'team44_LSANet.pth')
        model.load_state_dict(torch.load(model_path)["params_ema"], strict=True)
    elif model_id == 45:
        from models.team45_VRTDIP import DIPNet
        name, data_range = f"{model_id:02}_VRTDIP", 1.0
        model_path = os.path.join('model_zoo', 'team45_VRTDIP.pth')
        model = DIPNet()
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 46:
        from models.team46_ERRN import ERRN
        name, data_range = f"{model_id:02}_ERRN", 1.0
        model_path = os.path.join('model_zoo', 'team46_ERRN.pth')
        model = ERRN(in_channels=3,out_channels=3,feature_channels=40,upscale=4)
        model.load_state_dict(torch.load(model_path), strict=True)
    elif model_id == 48:
        from models.team48_EfficientSRGAN import RRDBNet
        name, data_range = f"{model_id:02}_EfficientSRGAN", 1.0
        # model_path = os.path.join('model_zoo', 'team48_EfficientSRGAN.pth')
        model_path = os.path.join('model_zoo', 'team48_EfficientSRGAN_update.pth')
        model = RRDBNet(3, 3, 64, 23, gc=32)
        model.load_state_dict(torch.load(model_path), strict=True)
    # elif model_id == 1:
    #     from models.team[Your_Team_ID]_[Model_Name] import [Model_Name]
    #     ...
    else:
        raise NotImplementedError(f"Model {model_id} is not implemented.")

    # print(model)
    model.eval()
    tile = None
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    return model, name, data_range, tile


def select_dataset(data_dir, mode):
    # inference on the DIV2K_LSDIR_test set
    if mode == "test":
        path = [
            (
                p.replace("_HR", "_LR").replace(".png", "x4.png"),
                p
            ) for p in sorted(glob.glob(os.path.join(data_dir, "DIV2K_LSDIR_test_HR/*.png")))
        ]

    # inference on the DIV2K_LSDIR_valid set
    elif mode == "valid":
        path = [
            (
                p.replace("_HR", "_LR").replace(".png", "x4.png"),
                p
            ) for p in sorted(glob.glob(os.path.join(data_dir, "DIV2K_LSDIR_valid_HR/*.png")))
        ]
    else:
        raise NotImplementedError(f"{mode} is not implemented in select_dataset")
    
    return path


def forward(img_lq, model, tile=None, tile_overlap=32, scale=4):
    if tile is None:
        # test the image as a whole
        output = model(img_lq)
    else:
        # test the image tile by tile
        b, c, h, w = img_lq.size()
        tile = min(tile, h, w)
        tile_overlap = tile_overlap
        sf = scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
        output = E.div_(W)

    return output

def forward_team34(img_lq, model, tile=None, tile_overlap=32, scale=4):
    if torch.cuda.is_available():
        img_lq = img_lq.cuda()
    window_size = 16
    scale = 4
    mod_pad_h, mod_pad_w = 0, 0
    _, _, h, w = img_lq.size()
    if h % window_size != 0:
        mod_pad_h = window_size - h % window_size
    if w % window_size != 0:
        mod_pad_w = window_size - w % window_size
    img = F.pad(img_lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

    model.eval()
    with torch.no_grad():
        output = model(img)

    _, _, h, w = output.size()
    output = output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]    
    
    return output

def run(model, model_name, data_range, tile, logger, device, args, mode="test"):

    sf = 4
    border = sf
    results = dict()
    results[f"{mode}_runtime"] = []
    results[f"{mode}_psnr"] = []
    if args.ssim:
        results[f"{mode}_ssim"] = []
    # results[f"{mode}_psnr_y"] = []
    # results[f"{mode}_ssim_y"] = []

    # --------------------------------
    # dataset path
    # --------------------------------
    data_path = select_dataset(args.data_dir, mode)
    save_path = os.path.join(args.save_dir, model_name, mode)
    util.mkdir(save_path)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for i, (img_lr, img_hr) in enumerate(data_path):

        # --------------------------------
        # (1) img_lr
        # --------------------------------
        img_name, ext = os.path.splitext(os.path.basename(img_hr))
        img_lr = util.imread_uint(img_lr, n_channels=3)
        img_lr = util.uint2tensor4(img_lr, data_range)
        img_lr = img_lr.to(device)

        # --------------------------------
        # (2) img_sr
        # --------------------------------
        start.record()
        if args.model_id == 34:
            img_sr = forward_team34(img_lr, model, tile)
        else:
            img_sr = forward(img_lr, model, tile)
        end.record()
        torch.cuda.synchronize()
        results[f"{mode}_runtime"].append(start.elapsed_time(end))  # milliseconds
        img_sr = util.tensor2uint(img_sr, data_range)

        # --------------------------------
        # (3) img_hr
        # --------------------------------
        img_hr = util.imread_uint(img_hr, n_channels=3)
        img_hr = img_hr.squeeze()
        img_hr = util.modcrop(img_hr, sf)

        # --------------------------------
        # PSNR and SSIM
        # --------------------------------

        # print(img_sr.shape, img_hr.shape)
        psnr = util.calculate_psnr(img_sr, img_hr, border=border)
        results[f"{mode}_psnr"].append(psnr)

        if args.ssim:
            ssim = util.calculate_ssim(img_sr, img_hr, border=border)
            results[f"{mode}_ssim"].append(ssim)
            logger.info("{:s} - PSNR: {:.2f} dB; SSIM: {:.4f}.".format(img_name + ext, psnr, ssim))
        else:
            logger.info("{:s} - PSNR: {:.2f} dB".format(img_name + ext, psnr))

        # if np.ndim(img_hr) == 3:  # RGB image
        #     img_sr_y = util.rgb2ycbcr(img_sr, only_y=True)
        #     img_hr_y = util.rgb2ycbcr(img_hr, only_y=True)
        #     psnr_y = util.calculate_psnr(img_sr_y, img_hr_y, border=border)
        #     ssim_y = util.calculate_ssim(img_sr_y, img_hr_y, border=border)
        #     results[f"{mode}_psnr_y"].append(psnr_y)
        #     results[f"{mode}_ssim_y"].append(ssim_y)
        # print(os.path.join(save_path, img_name+ext))
            
        # Save Restored Images
        # util.imsave(img_sr, os.path.join(save_path, img_name+ext))

    results[f"{mode}_memory"] = torch.cuda.max_memory_allocated(torch.cuda.current_device()) / 1024 ** 2
    results[f"{mode}_ave_runtime"] = sum(results[f"{mode}_runtime"]) / len(results[f"{mode}_runtime"]) #/ 1000.0
    results[f"{mode}_ave_psnr"] = sum(results[f"{mode}_psnr"]) / len(results[f"{mode}_psnr"])
    if args.ssim:
        results[f"{mode}_ave_ssim"] = sum(results[f"{mode}_ssim"]) / len(results[f"{mode}_ssim"])
    # results[f"{mode}_ave_psnr_y"] = sum(results[f"{mode}_psnr_y"]) / len(results[f"{mode}_psnr_y"])
    # results[f"{mode}_ave_ssim_y"] = sum(results[f"{mode}_ssim_y"]) / len(results[f"{mode}_ssim_y"])
    logger.info("{:>16s} : {:<.3f} [M]".format("Max Memory", results[f"{mode}_memory"]))  # Memery
    logger.info("------> Average runtime of ({}) is : {:.6f} milliseconds".format("test" if mode == "test" else "valid", results[f"{mode}_ave_runtime"]))
    logger.info("------> Average PSNR of ({}) is : {:.6f} dB".format("test" if mode == "test" else "valid", results[f"{mode}_ave_psnr"]))

    return results


def main(args):

    utils_logger.logger_info("NTIRE2024-EfficientSR", log_path="NTIRE2024-EfficientSR.log")
    logger = logging.getLogger("NTIRE2024-EfficientSR")

    # --------------------------------
    # basic settings
    # --------------------------------
    torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    json_dir = os.path.join(os.getcwd(), "results.json")
    if not os.path.exists(json_dir):
        results = dict()
    else:
        with open(json_dir, "r") as f:
            results = json.load(f)

    # --------------------------------
    # load model
    # --------------------------------
    model, model_name, data_range, tile = select_model(args, device)
    logger.info(model_name)

    # if model not in results:
    if True:
        # --------------------------------
        # restore image
        # --------------------------------

        # inference on the DIV2K_LSDIR_valid set
        valid_results = run(model, model_name, data_range, tile, logger, device, args, mode="valid")
        # record PSNR, runtime
        results[model_name] = valid_results

        # inference conducted by the Organizer on DIV2K_LSDIR_test set
        if args.include_test:
            test_results = run(model, model_name, data_range, tile, logger, device, args, mode="test")
            results[model_name].update(test_results)

        input_dim = (3, 256, 256)  # set the input dimension
        activations, num_conv = get_model_activation(model, input_dim)
        activations = activations/10**6
        logger.info("{:>16s} : {:<.4f} [M]".format("#Activations", activations))
        logger.info("{:>16s} : {:<d}".format("#Conv2d", num_conv))

        # The FLOPs calculation in previous NTIRE_ESR Challenge
        # flops = get_model_flops(model, input_dim, False)
        # flops = flops/10**9
        # logger.info("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))

        # fvcore is used in NTIRE2024_ESR for FLOPs calculation
        input_fake = torch.rand(1, 3, 256, 256).to(device)
        flops = FlopCountAnalysis(model, input_fake).total()
        flops = flops/10**9
        logger.info("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))

        num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
        num_parameters = num_parameters/10**6
        logger.info("{:>16s} : {:<.4f} [M]".format("#Params", num_parameters))
        results[model_name].update({"activations": activations, "num_conv": num_conv, "flops": flops, "num_parameters": num_parameters})

        with open(json_dir, "w") as f:
            json.dump(results, f)
    if args.include_test:
        fmt = "{:20s}\t{:10s}\t{:10s}\t{:14s}\t{:14s}\t{:14s}\t{:10s}\t{:10s}\t{:8s}\t{:8s}\t{:8s}\n"
        s = fmt.format("Model", "Val PSNR", "Test PSNR", "Val Time [ms]", "Test Time [ms]", "Ave Time [ms]",
                       "Params [M]", "FLOPs [G]", "Acts [M]", "Mem [M]", "Conv")
    else:
        fmt = "{:20s}\t{:10s}\t{:14s}\t{:10s}\t{:10s}\t{:8s}\t{:8s}\t{:8s}\n"
        s = fmt.format("Model", "Val PSNR", "Val Time [ms]", "Params [M]", "FLOPs [G]", "Acts [M]", "Mem [M]", "Conv")
    for k, v in results.items():
        val_psnr = f"{v['valid_ave_psnr']:2.2f}"
        val_time = f"{v['valid_ave_runtime']:3.2f}"
        mem = f"{v['valid_memory']:2.2f}"
        
        num_param = f"{v['num_parameters']:2.3f}"
        flops = f"{v['flops']:2.2f}"
        acts = f"{v['activations']:2.2f}"
        conv = f"{v['num_conv']:4d}"
        if args.include_test:
            # from IPython import embed; embed()
            test_psnr = f"{v['test_ave_psnr']:2.2f}"
            test_time = f"{v['test_ave_runtime']:3.2f}"
            ave_time = f"{(v['valid_ave_runtime'] + v['test_ave_runtime']) / 2:3.2f}"
            s += fmt.format(k, val_psnr, test_psnr, val_time, test_time, ave_time, num_param, flops, acts, mem, conv)
        else:
            s += fmt.format(k, val_psnr, val_time, num_param, flops, acts, mem, conv)
    with open(os.path.join(os.getcwd(), 'results.txt'), "w") as f:
        f.write(s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("NTIRE2024-EfficientSR")
    parser.add_argument("--data_dir", default="../", type=str)
    parser.add_argument("--save_dir", default="../results", type=str)
    parser.add_argument("--model_id", default=0, type=int)
    parser.add_argument("--include_test", action="store_true", help="Inference on the DIV2K_LSDIR test set")
    parser.add_argument("--ssim", action="store_true", help="Calculate SSIM")

    args = parser.parse_args()
    pprint(args)

    main(args)
