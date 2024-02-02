# BioMMNet
A Multimodal Network for Biological Data Analysis and Generation
```sh
BioMMNet/
│
├── data/                           # Directory for datasets
│   ├── preprocess.py               
│   ├── loader.py                   
│   └── ...                         # Other data-related scripts or modules
│
├── models/                         # Directory for model definitions
│   ├── base_model.py               
│   ├── sequence_model.py           
│   ├── image_model.py              
│   ├── fusion_model.py             
│   └── ...                         # Additional model-related scripts or modules
│
├── utils/                          # Utility tools directory
│   ├── metrics.py                  
│   ├── visualizations.py           
│   └── ...                         # Other utility scripts or helper functions
│
├── notebooks/                      # Jupyter notebooks directory
│   ├── exploration.ipynb           
│   └── ...                         
│
├── logs/                           
│                                   
├── checkpoints/                    
│                                   
├── config/                         # Configuration files directory
│   ├── config.yaml                 # Configuration file with parameters and settings
│   └── ...                         
│
├── requirements.txt                
├── bio_sequence_cli.py             # Example script for the project.
└── README.md                       
```
<!-- GETTING STARTED -->
## Getting Started

This is an sample example guides you on how to run project scripts locally by the following simple example steps.

<!-- USAGE EXAMPLES -->
## Usage
First, you need to download the checkpoint for [progen2-small](https://github.com/salesforce/progen/tree/main/progen2).
Or run the follwing code.
  ```sh 
  # checkpoint
  model=progen2-small
  wget -P checkpoints/${model} https://storage.googleapis.com/sfr-progen-research/checkpoints/${model}.tar.gz
  tar -xvf checkpoints/${model}/${model}.tar.gz -C checkpoints/${model}/
  ```


Then, you can run the bio_sequence_cli script. 

Simply input what you want the multimodal model to do. 

For example, "Please generate a protein sequence with length 64, and please set the temperature to 0.7"
  ```sh 
  python bio_sequence_cli.py "Please generate a protein sequence with length 64, and please set the temperature to 0.7" 
  ```
