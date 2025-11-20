# 1. import and load
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

# 2. train-test split
X=np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
y=np.array([0,0,0,0,1,1,1,1,1,1])

# 3. train the model
model=RandomForestClassifier(n_estimators=50,max_depth=3)
model.fit(X,y)

# 4. predict
pred=model.predict(X)

# 5. evaluate (accuracy + confusion matrix)
acc=accuracy_score(y,pred)
cm=confusion_matrix(y,pred)
precision=precision_score(y,pred)
recall=recall_score(y,pred)
f1=f1_score(y,pred)

print("---------------------Evaluation Metrics--------------------------")
print(f"Accuracy : {acc}\n Confusion Matrix : {cm}\n Precision : {precision}\n Recall : {recall}\n F1-Score : {f1}")

# 6. plotting
plt.figure(figsize=(10,6))

plt.scatter(X,y,color="red",label="Actual")
plt.plot(X,pred,color="blue",label="Predicted")

plt.title("RandomForest Classifier")
plt.xlabel("Hours")
plt.ylabel("Class")
plt.legend()
plt.grid(True)
plt.savefig("StudentSuccessPredictor.png")
plt.show()

# 6. print reflections
print("Initially i've imported all the required packages and frameworks, then loaded the data, then trained the model(Here i used RandomForestClassifier),then predicted/tested the model, then evaluated the model performance using the evaluation metrics such as accuracy, precison, recall, confusion matrix, etc and at the end i plotted the model prediction. This the complete ML model pipeline.:) ")
