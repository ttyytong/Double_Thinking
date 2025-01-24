---

# Double_Thinking

This project is developed based on the [CogVLM](https://github.com/xxx/CogVLM) framework, aiming to integrate Prompt Engineering and Double Thinking for open-set construction site safety inspection tasks.

## Features

This repository mainly includes the following modules:

1. **Basic Demo (basic_demo)**: Provides examples of basic usage.
2. **Composite Demo (composite_demo)**: Demonstrates more complex feature combinations.
3. **Data Processing (data_process)**: Includes tools for data cleaning and standardization.
4. **Finetune Demo (finetune_demo)**: Examples of model fine-tuning for specific tasks.
5. **OpenAI Demo (openai_demo)**: Extension examples using OpenAI APIs.
6. **Utilities (utils)**: General-purpose utility functions.

## Key Technologies

- **CogVLM Framework**: The base model framework that provides multimodal understanding and reasoning capabilities.
- **text2vec**: For semantic similarity assessment, supporting efficient text embedding and similarity calculation.
- **Standardization Tool**: Implements data standardization via the `standardization.py` file.

## Usage Guide

### Environment Setup

Install the necessary dependencies using either `environment.yml` or `requirements.txt`:

```bash
conda env create -f environment.yml
# Or
pip install -r requirements.txt
```

### Model Files

Model files and checkpoints can be downloaded from the following link:
[https://huggingface.co/ttyytong/Double_Thinking](https://huggingface.co/ttyytong/Double_Thinking)

Please extract the downloaded files and place them in the `checkpoints/` directory.

### Quick Start

1. Run the basic demo:

   ```bash
   python basic_demo/demo.py
   ```

2. Test semantic similarity:

   ```bash
   python text2vec_demo.py
   ```

3. Fine-tune the model:

   ```bash
   python finetune_demo/train.py
   ```

## Contribution

Contributions to this project are welcome! To submit issues or improvement suggestions, please open an issue or a pull request.

## License

This project is licensed under the [Apache 2.0 License](LICENSE).

## Citation & Acknowledgements

If you find our work helpful, please consider citing the following papers:

@misc{wang2023cogvlm,
title={CogVLM: Visual Expert for Pretrained Language Models},
author={Weihan Wang and Qingsong Lv and Wenmeng Yu and Wenyi Hong and Ji Qi and Yan Wang and Junhui Ji and Zhuoyi Yang and Lei Zhao and Xixuan Song and Jiazheng Xu and Bin Xu and Juanzi Li and Yuxiao Dong and Ming Ding and Jie Tang},
year={2023},
eprint={2311.03079},
archivePrefix={arXiv},
primaryClass={cs.CV}
}

@misc{hong2023cogagent,
title={CogAgent: A Visual Language Model for GUI Agents},
author={Wenyi Hong and Weihan Wang and Qingsong Lv and Jiazheng Xu and Wenmeng Yu and Junhui Ji and Yan Wang and Zihan Wang and Yuxiao Dong and Ming Ding and Jie Tang},
year={2023},
eprint={2312.08914},
archivePrefix={arXiv},
primaryClass={cs.CV}
}

