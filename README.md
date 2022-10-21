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

We firstly read all the images from the downloaded Zip File by using the zipfile library. Then we show 10 sample faces from the dataset using the matplotlib library. Following to this we print the details, the number of people detected and the total number of images and then iterate through the dataset and use smiling faces as the Training dataset. We select non-smiling faces for the testing data. 

**Training Data**

![index](https://user-images.githubusercontent.com/75626387/197199362-a9f59605-5ed1-40f0-bd72-23d913029ad0.png)

**Testing Data**

![index](https://user-images.githubusercontent.com/75626387/197199387-c0969a80-d8e9-4c12-803c-d706f3a387b4.png)


Now we create a NxM matrix with N images and M pixels per image, following which we import PCA from Scikit Learn Library in Python and apply PCA and take first K principal components as eigenfaces.

```bash
n_components = 2
eigenfaces = pca.components_[:n_components]

fig, axes = plt.subplots(2,sharex=True,sharey=True,figsize=(8,10))
for i in range(2):
    axes[i].imshow(eigenfaces[i].reshape(faceshape), cmap="gray")
plt.show()
```
![index](https://user-images.githubusercontent.com/75626387/197199788-0e923caa-c404-484e-9405-f05193030ee1.png)

Showing the sample eigenfaces generated from any 10 images.

![image](https://user-images.githubusercontent.com/75626387/197199964-0483f959-7f7f-469b-ab2c-6002c6706ed7.png)

Now, we Generate weights as a KxN matrix where K is the number of eigenfaces and N the number of samples.

```bash
# Generate weights as a KxN matrix where K is the number of eigenfaces and N the number of samples
weights = eigenfaces @ (facematrix - pca.mean_).T
print("Shape of the weight matrix:", weights.shape)
```


## Authors

- Aryaman : [@Aryaman-Arya](https://github.com/Aryaman-Arya)
- Sanjam Kaur Bedi : [@Sanjam-Bedi](https://github.com/Sanjam-Bedi)
- Vishal Nitnaware : [@vishal7474](https://github.com/vishal7474)

