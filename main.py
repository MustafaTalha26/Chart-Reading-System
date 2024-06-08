import linereading
import piereading
import barreading
import model

imagepath = 'testdatadata/Line/544.png'
index = model.predict_chart(imagepath)
if index == 0:
    barreading.horizontal_bar_graph(imagepath)
if index == 1:
    barreading.vertical_bar_graph(imagepath)
if index == 2:
    linereading.read_line_graph(imagepath)
if index == 3:
    piereading.read_pie_chart(imagepath)