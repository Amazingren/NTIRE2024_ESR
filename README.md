# [NTIRE 2024 Challenge on Efficient Super-Resolution](https://cvlai.net/ntire/2024/) @ [CVPR 2024](https://cvpr.thecvf.com/)

## About the Challenge

Jointly with NTIRE workshop we have a challenge on Efficient Super-Resolution, that is, the task of super-resolving (increasing the resolution) an input image with a magnification factor x4 based on a set of prior examples of low and corresponding high resolution images. The challenge has three tracks.

The aim is to devise a network that reduces one or several aspects such as runtime, parameters, FLOPs, activations, and depth of RLFN (https://arxiv.org/pdf/2205.07514.pdf), the winner solution of the NTIRE2022 Efficient Super-Resolution Challenge, while at least maintaining PSNR of 29.00dB on validation datasets and PSNR 28.72 on test set.

Note that for the final ranking and challenge winners we are weighing more the teams/participants improving in more than one aspect (runtime, parameters, FLOPs, activations, depths) over the provided reference solution.

For the sake of fairness, please do not train your model with the validation LR images, validation HR images, and testing LR images.

## The Environments

The evaluation environments adopted by us is recorded in the `requirements.txt`. After you built you own basic Python setup via either *virtual environment* or *anaconda*, please try to keep similar to it via:

```pip install -r requirements.txt```

## The Validation datasets

TODO
After downloaded all the necessary validate datasets, please organize them as follows:

```
|NTIRE2024_ESR_Challenge/
|--DIV2K_valid_HR/
|    |--0801.png
|    |--...
|    |--0900.png
|--DIV2K_valid_LR/
|    |--0801x4.png
|    |--...
|    |--0900x4.png
|--LSDIR_valid_HR/
|    |--0000001.png
|    |--...
|    |--0000100.png
|--LSDIR_valid_LR/
|    |--0000001x4.png
|    |--...
|    |--0000100x4.png
|--NTIRE2024_ESR/
|    |--...
|    |--test_demo.py
|    |--...
|--results/
```

## How to test the baseline model?

1. `git clone https://github.com/Amazingren/NTIRE2024_ESR.git`
2. Select the model you would like to test from [`run.sh`](./run.sh)
    ```bash
    CUDA_VISIBLE_DEVICES=0 python test_demo.py --data_dir [path to your data dir] --save_dir [path to your save dir] --model_id -1
    ```
    - Be sure the change the directories `--data_dir` and `--save_dir`.


## How to add your model to this baseline?

1. Register your team in the [Google Spreadsheet](https://docs.google.com/spreadsheets/d/1ZFlte0uR4bNl6UVJxShESkui1n3ejzXAvUX_e1qyhSc/edit?usp=sharing) and get your team ID.
2. Put your the code of your model in `./models/[Your_Team_ID]_[Your_Model_Name].py`
   - Please add **only one** file in the folder `./models`. **Please do not add other submodules**.
   - Please zero pad [Your_Team_ID] into two digits: e.g. 00, 01, 02 
3. Put the pretrained model in `./model_zoo/[Your_Team_ID]_[Your_Model_Name].[pth or pt or ckpt]`
   - Please zero pad [Your_Team_ID] into two digits: e.g. 00, 01, 02  
4. Add your model to the model loader `./test_demo/select_model` as follows:
    ```python
        elif model_id == [Your_Team_ID]:
            # define your model and load the checkpoint
    ```
   - Note: Please set the correct data_range, either 255.0 or 1.0
5. Send us the command to download your code, e.g, 
   - `git clone [Your repository link]`
   - We will do the following steps to add your code and model checkpoint to the repository.
   

## How to calculate the number of parameters, FLOPs, and activations

```python
    from utils.model_summary import get_model_flops, get_model_activation
    from models.team00_RLFN import RLFN_Prune
    from fvcore.nn import FlopCountAnalysis

    model = RLFN_Prune()
    
    input_dim = (3, 256, 256)  # set the input dimension
    activations, num_conv = get_model_activation(model, input_dim)
    activations = activations / 10 ** 6
    print("{:>16s} : {:<.4f} [M]".format("#Activations", activations))
    print("{:>16s} : {:<d}".format("#Conv2d", num_conv))

    flops = get_model_flops(model, input_dim, False)
    flops = flops / 10 ** 9
    print("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))

    # fvcore is also used in NTIRE2024_ESR for FLOPs calculation
    # flops = FlopCountAnalysis(model, input_dim).total()
    # flops = flops/10**9
    # logger.info("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))

    num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    num_parameters = num_parameters / 10 ** 6
    print("{:>16s} : {:<.4f} [M]".format("#Params", num_parameters))
```

## License and Acknowledgement
This code repository is release under [MIT License](LICENSE). 