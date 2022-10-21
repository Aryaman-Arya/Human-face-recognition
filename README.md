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

* **Training Data**

![index](https://user-images.githubusercontent.com/75626387/197199362-a9f59605-5ed1-40f0-bd72-23d913029ad0.png)

* **Testing Data**

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

![image](https://user-images.githubusercontent.com/75626387/197204118-7336e2c2-e0b7-4194-b9d1-ab6d6cd432de.png)

Now, we Generate weights as a KxN matrix where K is the number of eigenfaces and N the number of samples.

```bash
weights = eigenfaces @ (facematrix - pca.mean_).T
print("Shape of the weight matrix:", weights.shape)
```
## Results

* **Query- 1**

![image](https://user-images.githubusercontent.com/75626387/197204741-72401d8f-24fa-48f7-aa7d-8dee692eecdd.png)

* **Query- 2**

![image](https://user-images.githubusercontent.com/75626387/197204797-d95c0e35-fc91-4ca8-be0d-2c17a400ab42.png)

* **Query- 3**

![image](https://user-images.githubusercontent.com/75626387/197204840-c016d95c-8ed0-4f5f-b704-f3d2b51cac8e.png)

* **Query- 4**

![image](https://user-images.githubusercontent.com/75626387/197204915-ecb43d20-9c8a-4ac6-bfe4-c6a0dba96409.png)

* **Query- 5**

![image](https://user-images.githubusercontent.com/75626387/197204989-3832bde0-3b81-4c22-a9f4-f14d11738fc7.png)

## Visualizing the mean face and random face

![image](https://user-images.githubusercontent.com/75626387/197205063-93e570b9-6774-402c-b787-17d4e3f9217f.png)


## Authors

- Aryaman : [@Aryaman-Arya](https://github.com/Aryaman-Arya)
- Sanjam Kaur Bedi : [@Sanjam-Bedi](https://github.com/Sanjam-Bedi)
- Vishal Nitnaware : [@vishal7474](https://github.com/vishal7474)

