import linereading
import piereading
import barreading
import model

imagepath = 'testdata/BarH/301.png'
index = model.predict_chart(imagepath)
print(index)
if index == 0:
    barreading.horizontal_bar_graph(imagepath)
if index == 1:
    barreading.vertical_bar_graph(imagepath)
if index == 2:
    linereading.read_line_graph(imagepath,['en'],30,2)
if index == 3:
    piereading.read_pie_chart(imagepath)