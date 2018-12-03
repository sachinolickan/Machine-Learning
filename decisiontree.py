from sklearn.tree import DecisionTreeClassifier

x=[[10,'0'],[20,'1'],[13,'1'],[15,'1'],[18,'0']]
y=[['candycrush'],['whatsapp'],['facebook'],['instagram'],['game']]

clf=DecisionTreeClassifier()
clf.fit(x,y)

print("enter age and gender:")
d=input()

list1=list()
list1.append(d)

gender=input()
list1.append(gender)

p=clf.predict([list1])
print(p)
print("plays: ",p[0])