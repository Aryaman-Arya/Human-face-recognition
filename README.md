# Human face recognition for Identity theft protection using Principal Component Analysis

Implemented exploratory data analysis and dimensionality reduction on colossal media dataset using
Principal Component Analysis.




## Python Libraries 

* **[Pandas](https://pandas.pydata.org/docs/getting_started/install.html)** 

* **[NumPy](https://numpy.org/install/)** 

* **[Matplotlib](https://matplotlib.org/stable/users/installing/index.html)** 

* **[Seaborn](https://seaborn.pydata.org/installing.html)** 

* **[Scikit-learn](https://scikit-learn.org/stable/install.html)** 




## Implementation

We firstly read all the images from the downloaded Zip File by using the zipfile library. Then we show 10 sample faces from the dataset using the matplotlib library. Following to this we print the details, the number of people detected and the total number of images and then iterate through the dataset and use non-smiling faces as the testing data.

```bash
fig, axes = plt.subplots(5,2,sharex=True,sharey=True,figsize=(8,10))
faceimages = list(faces.values())[-10:] # take last 10 images
for i in range(10):
    axes[i%5][i//5].imshow(faceimages[i], cmap="gray")
plt.show()
```
###Image 

Now we create a NxM matrix with N images and M pixels per image, following which we import PCA from Scikit Learn Library in Python and apply PCA and take first K principal components as eigenfaces.

```bash
n_components = 2
eigenfaces = pca.components_[:n_components]

fig, axes = plt.subplots(2,sharex=True,sharey=True,figsize=(8,10))
for i in range(2):
    axes[i].imshow(eigenfaces[i].reshape(faceshape), cmap="gray")
plt.show()
```

##Image 2


## Authors

- Aryaman : [@Aryaman-Arya](https://github.com/Aryaman-Arya)
- Sanjam Kaur Bedi : [@Sanjam-Bedi](https://github.com/Sanjam-Bedi)
- Vishal Nitnaware : [@vishal7474](https://github.com/vishal7474)

