# Ethos: Rectifying Language Models in Orthogonal Parameter Space
Code repository for the paper:

"[Ethos: Rectifying Language Models in Orthogonal Parameter Space](https://arxiv.org/abs/2403.08994)"

## Code Structure
For each unlearning task, there is a dedicated folder named after the task. These folders contain all necessary datasets and training/evaluation code for conducting experiments related to that specific task. Within each task folder, there is also a collection of scripts for different models. 

## Getting Started

1. Create a virtual environment: 
    ```
    conda create -n ethos python=3.9
    ```

2. Install required python packages:
    ```
    pip install -r requirements.txt
    ```

3. Navigate to the corresponding folder and run the provided script. For instance, to perform the toxification unlearning task for the OPT-1.3B model: 
    ```
    cd toxification
    sh opt-1.3b.sh
    ```
* These scripts first fine-tune the model on specific datasets to generate task vectors. They then produce a task vector for the unlearning purpose. Finally, the scripts evaluate the model's unlearning performance after incorporating the task vector. Please refer to the scripts for detailed execution commands.
* To run experiment on Llama model, make sure you have requested access in the official [Meta Llama-2](https://huggingface.co/meta-llama/Llama-2-7b-hf) webpage on HuggingFace and have logged into your HuggingFace account with the access token. 

    ```
    huggingface-cli login
    ```

## Acknowledgement
This repo refers to the following projects:
* [stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca)
* [bias-bench](https://github.com/McGill-NLP/bias-bench)
* [Ext-Sub](https://github.com/YanshekWoo/Ext-Sub)
* [controlling-llm-memorization](https://github.com/amazon-science/controlling-llm-memorization)

## How to Cite
```
@inproceedings{
    gao2024ethos,
    title={Ethos: Rectifying Language Models in Orthogonal Parameter Space},
    author={Lei Gao and Yue Niu and Tingting Tang and Salman Avestimehr and Murali Annavaram},
    booktitle={2024 Annual Conference of the North American Chapter of the Association for Computational Linguistics},
    year={2024},
    url={https://arxiv.org/abs/2403.08994}
}
```

## Contact

Questions or comments can be sent to "leig AT usc.edu".
