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

```bash
query = faces["ns1/face1.pgm"].reshape(1,-1)
query_weight = eigenfaces @ (query - pca.mean_).T
print(query_weight.shape)
euclidean_distance = np.linalg.norm(weights - query_weight, axis=0)
print("Now to find best match !")
best_match = np.argmin(euclidean_distance)
print("Best match %s with Euclidean distance %f" % (facelabel[best_match], euclidean_distance[best_match]))
# Visualize
fig, axes = plt.subplots(1,2,sharex=True,sharey=True,figsize=(8,6))
axes[0].imshow(query.reshape(faceshape), cmap="gray")
axes[0].set_title("Query")
axes[1].imshow(facematrix[best_match].reshape(faceshape), cmap="gray")
axes[1].set_title("Best match")
plt.show()
```
_ _ (similar code for queries 2, 3, 4, 5) _ _

![image](https://user-images.githubusercontent.com/75626387/197205180-43f993b1-dc45-4bf6-aca1-89b49b3a2292.png)

* **Query- 2**

![image](https://user-images.githubusercontent.com/75626387/197205326-d7dd188f-a63b-4cbf-a04e-5350a58d8260.png)

* **Query- 3**

![image](https://user-images.githubusercontent.com/75626387/197205428-f1b324f1-dace-4b51-a234-d135cb8be54e.png)

* **Query- 4**

![image](https://user-images.githubusercontent.com/75626387/197205553-a69f4246-4cbe-44f4-9b55-562eb238dcf0.png)

* **Query- 5**

![image](https://user-images.githubusercontent.com/75626387/197205622-9538bda4-b752-4f3f-b0c8-c16c2a9cab0a.png)

## Visualizing the mean face and random face

![image](https://user-images.githubusercontent.com/75626387/197205687-d7b22bf7-b388-4fda-ac5b-c1a211d6a855.png)

By experimenting values of Principal Components from 1 to 10, we find that values below 5 predict with 90% accuracy and values above 5 give 100% accurate results. PC = 5 gives least Euclidean Distances on all faces.

## Authors

- Aryaman : [@Aryaman-Arya](https://github.com/Aryaman-Arya)
- Sanjam Kaur Bedi : [@Sanjam-Bedi](https://github.com/Sanjam-Bedi)
- Vishal Nitnaware : [@vishal7474](https://github.com/vishal7474)

