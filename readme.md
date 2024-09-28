this project includes code for web scraping (scrapingv2.py), preprocessing(gui_ver2.py), and the final notebook containing classifiers(notebook6.ipynb).

note that there is already a dataset contained within the shoeproj folder that was created using these scripts, so
you dont have to run all programs to formulate the dataset and can instead run the notebook. To access this dataset, please download the code zip attached to the one drive as the dataset file was too large to include, hence this folder doesnt contain dataset entries.

To run the notebook, the key things to be aware of this the directorys for CNN data are hard coded, so they must be modified for
your enviroment. Moreover, note that we use a noSQL mongoose database to fetch extracted features. To this end i have supplied my key to
access the database, thus you should be able to access it without issues. Finally ensure that all dependencies are installed.

When running the models, it may be neccesary to refetch the data, as the dataset may be altered for each models use. Hence to ensure
proper functionality, regenerate the dataset. The notebook follows the journey in which i used to generate my results, so hopefully its
easy to follow the steps till the classification outcomes.



