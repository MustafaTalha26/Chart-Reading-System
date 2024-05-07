import linereading
import piereading
import model

imagepath = 'rawdata/Line/99767.png'
index = model.predict_chart(imagepath)
print(index)
if index == 0:
    print("Barh")
if index == 1:
    print("BarV")
if index == 2:
    linereading.read_line_graph(imagepath,['en'],30)
if index == 3:
    piereading.read_pie_chart(imagepath)