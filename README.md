
**SUMMARY:**

After importing the dataset & required libraries and splitting the dataset into training and testing halves , I worked with my model of choice (Decision Tree) based on data characteristics and implemented it using sklearn library. I later created graphs for data visualization using seaborn . Then I finally created an endpoint that takes in the features (petal length, petal width, sepal length, sepal width) and sends back the classification result in a formatted way. Output can be seen at "http://127.0.0.1:5000/ " .


**1. Importing Libraries and Splitting the Dataset **

Before starting working on the dataset, I had to first download all the necessary python libraries- numpy, pandas, matplotlib, sklearn and seaborn and imported them into my notebook.

I was given the Iris Dataset which I downloaded, saved in comma-separated value format and read its values in a pandas DataFrame using the read_csv() function by supplying the file address.

Next, in order to check if there were NaN values or not I used the isnull().sum() function which returns the number of missing values in the data set in a column-wise format. As there were no missing values, I moved on further where I defined a DataFrame cdf containing the attributes -petal length, petal width, sepal length, sepal width and the DataFrame edf containing the Class of the flower as columns.  

As sklearn does not work directly with pandas DataFrames and TimeSeries so we need to convert it to numpy arrays first. Here x and y are numpy array versions of cdf and edf respectively.

In order to split the given dataset into testing and training part ,I used the train_test_split function from sklearn 's model_selection module which randomly divides the dataset into two parts. I selected test_size as 0.4 . I did not go for too big values of test_size considering how it affects the accuracy of my model or too small values as it might result in possible over-fitting of the model which I did not want.
In this split I set the random_state value as fixed integer which means that each time this function is called it will return the same set of train and test data points.


**2.  Selecting the appropriate method (Decision Tree Model)**

I decided to go with "Decision Tree Model" because when I visualized the data using seaborn I saw that some attributes showed clear demarcation between classes when I used them as classifiers (for example. Iris-setosa' could be clearly demarcated from the rest two species on the basis of petal length).

What are decision tree models and how do they work ?
Well, it is a supervised (teaching the model using labelled data ) ML Classification model which means it categorizes unknown items into different classes .It works by selecting attributes and then splitting the data based on the results of the attribute we tested in order to get more predictive classification results .
 
For example, in our dataset, we have four attributes sepal length, sepal width, petal length, petal width . Suppose we classify our dataset on the basis of attribute sepal length such that the flowers with sepal length belonging in range (a, b) belong to Class 'Iris-verginica' for 50 percent of the cases, those with sepal length belonging in range (c, d) belong to Class 'Iris-versicolor' for 60 percent of the cases, those with sepal length belonging in range (e, f) belong to Class 'Iris-setosa' for 55 percent of the cases. In this case our attribute selection is not very good because we cannot clearly specify the class of a particular flower when say its sepal length value belongs to one of the three above ranges.

In another example, suppose we classify our dataset on the basis of attribute petal width such that the flowers with petal width belonging in range (a', b') belong to Class 'Iris-verginica' for 90 percent of the cases, those with petal width belonging in range (c', d') belong to Class 'Iris-versicolor' for 60 percent of the cases, those with petal width belonging in range (e', f') belong to Class 'Iris-setosa' for 95 percent of the cases. In this scenario, our attribute selection is better than the previous one because we can predict the class of a particular flower very accurately when say its petal width value belongs to range (a', b') or range (e', f'). We say the nodes obtained after classification are more pure than in previous case as they are composed of mostly one of the classes. However, for petal width values belonging to the range (c', d'), we cannot comment confidently on the class of the flower, so for this set, we will again perform partitioning  with a different attribute so as to get more pure nodes after split.

So, the Decision Tree Classifier performs recursive partitioning where it selects attributes that give more pure nodes. The impurity of nodes or randomness in nodes is measured as its entropy. Suppose an attribute used as a classifier gives us equal contribution of all three classes for a particular range of values of that attribute, then its entropy would be 1. On the other hand if it gives 100% contribution from only 1 class then its entropy would be 1. The model seeks to decrease the entropy after each split which is done by selecting attributes that increase the information gain after a split where the information gain(E) is defined as the  entropy before the split - entropy after the split.


****	3. Implementing the Decision Tree Model****

Next I created an object Classifier using the function DecisionTreeClassifier() imported from sklearn library and trained this object Classifier using the fit() method with parameters criterion as 'entropy' and max_depth as 3. I set criterion as entropy as I wanted the model to make splits based on information gain at each step to get purer nodes that will ultimately make for a better model. I selected the max_depth (which is the maximum number of times the split will be performed before we get completely pure nodes ) as 3. Too high values of this parameter may result in over-fitting of the model (which means it will have poor out-of-sample cases accuracy) and too low values of this parameter may otherwise decrease the accuracy of our model as the model might not be able to capture enough information about the data. Hence, we need to come up with an optimal value for this parameter based on the testing data accuracy.

Now ,I imported the metrics module from sklearn and calculated accuracy_score which comes out to be 0.95 here (1 being best and 0 being worst). The accuracy_score is calculated as the number of intersection points in predicted_class array and the actual class array of the dataset divided by the total number of entities. In our case, say if we have 100 data points then our model correctly predicts the class of the flower with given parameters in 95 cases.


****4. Creating Data Visualizations using Seaborn ******

Now to visualize the data I imported the seaborn library.
 
sns.relplot(x="sepal length",y='Class',data=df,kind="scatter",size="sepal width")     sns.relplot(x="petal length",y='Class',data=df,kind="scatter",hue="sepal length",size="petal width")

I plotted two scatter plots - first was of class against sepal length with size as sepal width which means that the points in the plot will be scattered according to the correlation of sepal length of flowers belonging to a specific class with the size of the scatter dots proportional to the sepal width of flower. Similarly , in the second scatter plot where I added the 'hue' attribute which controls the shade of the scatter dots in the plot. The scatter plot reveal that 'Iris-setosa' can be clearly demarcated from the rest two species on the basis of petal length but not sepal length.

sns.catplot(x="petal width",y='Class',data=df,hue="petal length")

Then I made a catplot of class against petal width to find the categorical distribution of flowers belonging to a class with petal length as the shade. This tells us the general picture that flowers belonging to class 'Iris-setosa' have smaller values of petal width and petal length, those belonging to 'Iris-verginica' have higher values of petal width and petal length whereas those belonging to 'Iris-versicolor' have intermediate values of petal width and petal length.

sns.barplot(y="sepal length",x="Class",data=df)
sns.boxplot(x="sepal width",y="Class",data=df)
sns.swarmplot(x="petal width",y="Class",data=df)

From the bar plot of sepal length vs Class we can find that the flowers belonging to the class 'Iris-virginica' have highest sepal length followed by 'Iris-versicolor' and 'Iris-setosa'. The boxplot reveals us that there the  sepal width of 'Iris-setosa' species is highest with clear demarcation on the basis of sepal width from 'Iris-virginica' and 'Iris-Versicolor'.
The swarm plot tells us that although the average value of petal width in 'Iris-setosa' is lowest and 'Iris-verginica' is highest and 'Iris-versicolor' is in between the two and we can demarcate 'Iris-setosa' from rest on the basis of petal width directly but we cannot 
Separate 'Iris-verginica' and 'Iris-versicolor' from the rest directly on the basis of petal width.
 
sns.countplot(x="Class",data=df,hue="sepal width")
sns.countplot(x="Class",data=df,hue="petal width")

The count plot of sepal width tells us the number density of a particular sepal width in every class. For 'Iris-setosa' it is highest for sepal width value 3.7 and for 'Iris-versicolor' it is highest near sepal width value 3.0 and for 'Iris-virginica' it is highest for sepal width close to 2.9.Similarly, the count plot of petal width tells us that for 'Iris-setosa' the maxima is at petal width value 0.2 and for 'Iris-versicolor' it is highest near petal width value 1.2 and for 'Iris-virginica' it is highest for petal width close to 1.8. We conclude that that, if in a count plot, a new flower's attributes are close to the value of the attribute for which we get maximum number of flower count in a particular class, then the new flower may belong to that particular class, although it is not entirely necessary that it will. 


****5. Implementing 'REST API'****

In the beginning I imported the 'flask' which is a web framework to create websites. From flask I had to import the 'Flask' Class which will create an instance of the web application  for us and 'jsonify' constructor which will ensure that the output returned from our code is in 'JSON' serializable format as the web API 's deal with JSON or XML format only.

app = Flask(__name__)

Here app is the name of the instance of the flask app with " __name__" as a variable that gets assigned as "__main__" later when we run our code. Next I copied the ML model code part from my python notebook into this .py file which will be used later to return the predicted class of the flower when we enter its attributes through the endpoint. 

@app.route('/')
def hello_world():
    return jsonify("Hey User.....")

Next I defined a function hello_world that returns a message for the user to enter sepal length, sepal width, petal length, petal width after slashes to the Base URL "http://127.0.0.1:5000/ " to get the predicted class. Just above this function I mapped the function to the URL '/' which is the default URL and our message "Hey User....." will appear in json format only when we access the base URL.

@app.route('/<float:a>/<float:b>/<float:c>/<float:d>')
def Classifier2(a,b,c,d):
    result={
        "sepal length":a, 
        "sepal width":b, 
        "petal length":c,
        "petal width":d, 
        "Predicted Class":Class_tree.predict(a,b,c,d)[0] 
        }
    return result    

This function Classifier2 takes input four floating point numbers a,b,c,d from the endpoint when the user enters 4 real numbers separated by slashes in the URL.
The predicted class is displayed using the Class_tree_predict(a,b,c,d)[0] where the Class_tree_predict(a,b,c,d) returns the array containing the Class of the flower with given attributes. The complete information of the flower is stored in a python dictionary named result. Here we do not need to use jsonify function to convert the result as the result returned is a python dictionary which is already in JSON serializable format. 

if __name__=="__main__":
    app.run(debug="true") 

This is the part of code where the app created is run in development mode as we have set debug to "true" which will help us in tracing our errors.                                          


**** 6. Working with the endpoint *****
	
After running this file when we navigate to "http://127.0.0.1:5000/ " we can see the message 

   "Hey User, Enter the sepal length , sepal width , petal length , petal width as floating point numbers after slashes after    the base URL. Example enter /5.3/3.3/1.0/1.0 after base URL"

        From <http://127.0.0.1:5000/> 

is displayed and when we navigate to "http://127.0.0.1:5000/a/b/c/d" where a and b are floating point numbers , the sepal length, sepal width, petal length, petal width along with the predicted class is displayed as a python dictionary. When i enter some random float values of a,b,c,d the output is -

{
Predicted Class: "Iris-setosa",
petal length: 1.45,
petal width: 0.18,
sepal length: 5.67,
sepal width: 2.13
}

From <http://127.0.0.1:5000/5.67/2.13/1.45/0.18> 


Note : Open the web API after adding a JSON-view extension to your browser so the JSON document will look better formatted and with parts of the text highlighted.
![image](https://user-images.githubusercontent.com/75172544/123153776-b48ee500-d483-11eb-92e0-33d29d1df664.png)
