<div align="center">
  <h1>Compare Model</h1>
</div>

<h1>Overview</h1>
Model Compare is a tool to perform the model and calculate its mean average precision on given test dataset, in a simple and efficient way. 

<h1>Topics</h1>
<ul>
  <li><a href="#getting-started">Getting Started</a></li>
  <li><a href="#prerequisites">Prerequisites</a></li>
  <li><a href="#installation">Installation</a></li>
  <li><a href="#usage">Usage</li>
  <li><a href="#license">License</a></li>
  <li><a href="contribution">Contribution</a></li>
  <li><a href="#contact">Contact</a></li>
</ul>

<h2 id = "getting-started">Getting Started</h2>
These instructions below is going to help you get, run and improve it on your local machine for development and testing purposes right after training computer vision models.

<h2 id = "prerequisites">Prerequisites</h2>
Make sure you have the following installed before proceeding:

- [Python](https://www.python.org/) (version >= 3.x)
- [Supervision](https://github.com/roboflow/supervision)
- [Pip](https://pip.pypa.io/en/stable/installation/)
- [Virtualenv](https://virtualenv.pypa.io/)

* **Note:** Note that according to supervision library they only support images in `.jpg`, `.png`, `.jpeg` formats. Also they must be lowercase (such as, `example.png` not `example.PNG`). Please modify your images according to these information. 
  

<h2 id = "installation">Installation</h2>

```bash
# clone the repository
git clone https://github.com/mapilio/compare-model.git

# jump into the project directory
cd compare-model

# Create a virtual environment (in case you don't have virtualenv package please use `pip install virtualenv` to install it. If you don't want to install it then you may use `python -m venv compare-model-venv` as well. 
virtualenv compare-model-venv

# Activate the virtual env (on windows use `source compare-model-venv\.Scripts\activate`)
source compare-model-venv/bin/activate 

# Install dependencies
pip install -r requirements.txt
```

<h2 id = "usage">Usage</h2>

* Config takes all of your configurations to use model compare tool
```bash
python main.py --config config.yaml
```

* Config file arguments to configure configuration parameters
```bash
model_name: yolov5 # trained model name  (for now the tool is only compatible with yolov5 and yolov8, so give trained models with 'yolov5' or 'yolov8'. 
model_path: "example_model.pt" # trained model weight
image_path: "/images" # ground truth images path to validate trained model
project_name: "example-model-v-x" # trained project name 
project_folder_name: "example-model" # project folder name
conf_thresh: 0.5 # confidence threshold for model prediction
write_results: False # to decide to save prediction results or not 
calculate_map: True # whether to choose calculate mean average precision or not
image_size: 1280 # to set image size according to your model
annotation_path: "/ground_truth/labels" # ground truth labels path to validate trained model
yaml_path: "/cfg/example.yaml" # trained model's yaml file
act_mask: False # if your model provides masks 
task_mode: "detection" # to choose wheter to perform model on detection mode or segmentation mode 
verbose: True # to decide whether to see logs of predictions or not
```

<h2 id ="license">LICENSE</h2>
<p>This project is licensed under the MIT LICENSE - see the <code>LICENSE.md</code> file for details.</p>

<h2 id ="contribution">Contribution</h2>
<p>To make a contribution feel free to fork the repository, improve the project and then open a feature request.</p>

<h2 id="contact">Contact</h2>

For model compare tool's bug reports and feature requests please visit [GitHub Issues](https://github.com/mapilio/model-compare/issues), and join our [Discord](https://discord.com/invite/St5z2sUZ7H) community for questions and discussions!


