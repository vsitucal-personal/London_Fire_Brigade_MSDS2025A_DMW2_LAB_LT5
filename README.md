DMW2 LAB 2025A LT5
***
# London Fire Brigade Incident Records
![](images/title.png)
- [View notebook](DMW2%20LAB%20final.ipynb)
- [Latest version of dataset found here](https://data.london.gov.uk/dataset/london-fire-brigade-incident-records)

In this lab project the team has done feature analysis on on this dataset by using different clustering methods:
- Representative Based
  - K-Means
- Agglomerative/Heirarchical Based
  - Single Linkage
  - Average Linkage
  - Complete Linkage
  - Ward's Method
- Density-Based
  - DBSCAN
  - OPTICS

# Setup Environment
- Download [python](https://www.python.org/downloads/) at and run installation
- Make sure your python is in your `$PATH`
- Clone this repo to an ideal directory
- Setup environment in directory
```commandline
python -m venv venv
venv/Scripts/activate
pip install jupyter
pip install -r req.txt
```
- [Download lfb.db and put it in your directory](https://drive.google.com/file/d/1ht4k4X6CA783X2Un-42USxAF50qF-iL4/view?usp=sharing)
- [Download london borough shape file and unzip in your directory](https://drive.google.com/file/d/15_tdg5-zsn4dzFbYrlXdBQFvOC8AbScH/view?usp=sharing)

# Starting Up Jupyterlab
```commandline
jupyter lab
```