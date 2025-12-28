import sys
import playground
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QSlider, QSizePolicy
)
from PyQt5.QtCore import (Qt, QPoint)
from PyQt5.QtGui import (QPainter, QPen, QImage, QPixmap)
import numpy as np
#kullanıcı arayuzu
class StartScreen(QWidget):#açılan ilk ekran, başlat ekranı
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rakam Tanıma")
        self.resize(600,500)
        self.setStyleSheet("background-color:#A0DCF2")
        main_layout=QVBoxLayout()
        main_layout.setAlignment(Qt.AlignCenter)
        self.setLayout(main_layout)
        #başlık
        title_label=QLabel("RAKAM TANIMA UYGULAMASI")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet(""" 
           font-size:26px;
           font-weight:bold;
           color:#050000;                                             
        """)
        main_layout.addWidget(title_label)
        #robot resmi için
        image_label=QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        pixmap=QPixmap("robot.png")
        pixmap=pixmap.scaled(300,300,Qt.KeepAspectRatio,Qt.SmoothTransformation)
        image_label.setPixmap(pixmap)
        main_layout.addWidget(image_label)
        #başlatma butonu
        start_btn=QPushButton("Başlat")
        start_btn.setFixedSize(160,45)
        start_btn.setStyleSheet(""" 
           QPushButton{ 
             background-color:#F2FA98;
             font-size:16px;
             font-weight:bold;
             border-radius:10px;
           }
           QPushButton:hover{
             background-color:#90B7EE;
           }                                                                  
         """)
        start_btn.clicked.connect(self.start_app)
        main_layout.addWidget(start_btn,alignment=Qt.AlignCenter)#baslat butonu ortalanacak
    def start_app(self):
            self.main=MainWindow()#mainWindowa yönlendirecek
            self.main.show()
            self.close()    

class MainWindow(QMainWindow):
    def save_drawing(self):  # çizimi kaydet + playground ile tahmin et + matplotlib debug aç
        img = self.canvas.get_image()
        img.save("input.png", "PNG")  # debug için kalsın

        pred, probs = playground.predict_digit("input.png")

        self.prediction_label.setText(f"Tahmin: {pred}")
        self.confidence_label.setText(f"Güven: %{float(np.max(probs)) * 100:.2f}")

        # matplotlib penceresi açılsın:
        playground.show_debug("input.png", probs)

    def qimage_to_numpy(self, qimg):
        qimg = qimg.convertToFormat(QImage.Format_Grayscale8)#qimage'ı gri tonlamaya çevirme
        width = qimg.width()
        height = qimg.height()
        ptr = qimg.bits()
        ptr.setsize(height * width)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width))
        return arr
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rakam Tanıma")
        self.resize(900, 500)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        central_widget.setStyleSheet("background-color: #EDDCF2;")
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        self.canvas=DrawingCanvas()#çizim alanı
        self.canvas.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)#ekran büyüdükçe canvas da büyüyecek
        predict_btn=QPushButton("Tahmin et")#tahmin et butonu
        predict_btn.clicked.connect(self.save_drawing)#butona basınca png oluşacak
        predict_btn.setFixedHeight(45)
        predict_btn.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)#buton yatay olarak büyüyebilir ama dikey olarak büyüyemez
        predict_btn.setStyleSheet("background-color:#BFD8B8; color:white; font-size:16px; font-weight:bold; border-radius:8px;")
        center_layout = QVBoxLayout()
        center_layout.addWidget(self.canvas)
        center_layout.addWidget(predict_btn)
        center_layout.setStretch(0,10)
        center_layout.setStretch(1,0)
        center_layout.setContentsMargins(0,0,0,0)
        center_layout.setSpacing(10)
        center_widget=QWidget()
        center_widget.setLayout(center_layout)
        main_layout.setSpacing(10)
        self.prediction_label=QLabel("Tahmin: -")
        self.prediction_label.setAlignment(Qt.AlignCenter)
        self.prediction_label.setStyleSheet("font-size:24px; font-weight:bold; padding:20px; background-color:#ADC1DE; border:2px solid #A3C4BC; border-radius:8px;")
        self.confidence_label = QLabel("Güven: -")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        self.confidence_label.setStyleSheet("font-size:14px; padding:10px; background-color:#E2F0CB; border:2px solid #A3C4BC; border-radius:8px;")
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.prediction_label)
        right_layout.addWidget(self.confidence_label)
        right_layout.addStretch()
        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        right_widget.setFixedWidth(220)                                               
        controls_layout = QVBoxLayout()
        controls_layout.addWidget(QLabel("Kalem Renkleri"))
        black_btn=QPushButton("Siyah")
        black_btn.clicked.connect(lambda:self.canvas.set_pen_color(Qt.black))
        controls_layout.addWidget(black_btn)
        red_btn=QPushButton("Kırmızı")
        red_btn.clicked.connect(lambda:self.canvas.set_pen_color(Qt.red))
        controls_layout.addWidget(red_btn)
        blue_btn=QPushButton("Mavi")
        blue_btn.clicked.connect(lambda:self.canvas.set_pen_color(Qt.blue))
        controls_layout.addWidget(blue_btn)
        green_btn=QPushButton("Yeşil")
        green_btn.clicked.connect(lambda:self.canvas.set_pen_color(Qt.green))
        controls_layout.addWidget(green_btn)
        pink_btn=QPushButton("Pembe")
        pink_btn.clicked.connect(lambda:self.canvas.set_pen_color(Qt.magenta))
        controls_layout.addWidget(pink_btn)
        #butonlara renk ekleme
        black_btn.setStyleSheet("background-color:#4B4B4B; color:white; border-radius:5px; padding:5px; font-weight:bold;")
        red_btn.setStyleSheet("background-color:#FA7070; color:white; border-radius:5px; padding:5px; font-weight:bold;")
        blue_btn.setStyleSheet("background-color:#AEC6CF; color:white; border-radius:5px; padding:5px; font-weight:bold;")
        green_btn.setStyleSheet("background-color:#B5EAD7; color:white; border-radius:5px; padding:5px; font-weight:bold;")
        pink_btn.setStyleSheet("background-color:#FA70B3; color:white; border-radius:5px; padding:5px; font-weight:bold;")
        controls_layout.addSpacing(15)
        #kalem-silgi geçişi
        controls_layout.addWidget(QLabel("Araçlar"))
        tool_layout=QHBoxLayout()
        pen_btn=QPushButton("Kalem")
        eraser_btn=QPushButton("Silgi")
        pen_btn.clicked.connect(lambda:setattr(self.canvas,"current_tool","pen"))
        eraser_btn.clicked.connect(lambda:setattr(self.canvas,"current_tool","eraser"))
        tool_layout.addWidget(pen_btn)
        tool_layout.addWidget(eraser_btn)
        pen_btn.setStyleSheet("background-color:#FBE7C6; border:1px solid #E0D8C3; border-radius:5px; padding:5px; font-weight:bold;")
        eraser_btn.setStyleSheet("background-color:#FBE7C6; border:1px solid #E0D8C3; border-radius:5px; padding:5px; font-weight:bold;")
        controls_layout.addLayout(tool_layout)
        #tümünü temizle
        clear_btn=QPushButton("Tümünü temizle")
        clear_btn.clicked.connect(self.canvas.clear)
        controls_layout.addWidget(clear_btn)
        clear_btn.setStyleSheet("background-color:#EDADE4; color:white; border-radius:5px; padding:5px; font-weight:bold;")
        controls_layout.addWidget(QLabel("Kalınlık"))
        slider = QSlider(Qt.Horizontal)#yatay slider    
        slider.setMinimum(1)
        slider.setMaximum(40)
        slider.setValue(16)
        slider.valueChanged.connect(self.canvas.set_pen_width)
        slider.setStyleSheet("""
           QSlider::groove:horizontal { height:8px; background:#DDD; border-radius:4px; }
           QSlider::handle:horizontal { background:#AEC6CF; border:1px solid #B5EAD7; width:18px; margin:-5px 0; border-radius:9px; }
        """)
        controls_layout.addWidget(slider)
        controls_layout.addStretch() 
        controls_widget=QWidget()
        controls_widget.setLayout(controls_layout)
        controls_widget.setFixedWidth(200)
        main_layout.addWidget(controls_widget)
        main_layout.addWidget(center_widget,stretch=1)
        main_layout.addWidget(right_widget)
 
class DrawingCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(400,400)
        #arkaplanda duracak olan image
        self.image=QImage(400,400,QImage.Format_RGB32)
        self.image.fill(Qt.white)#silgi
        #kalem
        self.pen_color=Qt.black
        self.pen_width=16
        self.current_tool="pen"
        self.drawing=False #mouse basılı mı kontrolu
        self.last_point=None 
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            x_ratio=self.image.width()/self.width()
            y_ratio=self.image.height()/self.height()
            self.last_point = QPoint(
                int(event.pos().x()*x_ratio),
                int(event.pos().y()*y_ratio)
        )
    def mouseMoveEvent(self,event):
        if self.drawing:
            x_ratio = self.image.width() / self.width()
            y_ratio = self.image.height() / self.height()
            pos = QPoint(int(event.pos().x() * x_ratio), int(event.pos().y() * y_ratio))
            painter = QPainter(self.image)
            pen = QPen(self.pen_color if self.current_tool == "pen" else Qt.white,
                   self.pen_width, Qt.SolidLine, Qt.RoundCap)
            painter.setPen(pen)
            painter.drawLine(self.last_point, pos)
            self.last_point = pos
            self.update()         
    def mouseReleaseEvent(self,event):
        if event.button()==Qt.LeftButton:#çizgi birleşmesini engeller
            self.drawing=False
            self.last_point=None 
    def paintEvent(self,event):# ekrana çizimi bastırma
        canvas_painter=QPainter(self)
        canvas_painter.drawImage(self.rect(),self.image) 
    def set_pen_color(self,color):
        self.pen_color=color
        self.current_tool="pen"
    def set_pen_width(self,width):
        self.pen_width=width
    def use_eraser(self):
        self.current_tool="eraser"  
    def clear(self):#tümünü silme
        self.image.fill(Qt.white)
        self.update()#ekranı yeniden çizer
    def save_png(self,filename="input.png"):#çizimi png olarak kaydet
        self.image.save(filename,"PNG")
    def get_image(self):#çizimi qimage olarak döndür
        return self.image
                                               

if __name__ == "__main__":
    app = QApplication(sys.argv)
    start=StartScreen()
    start.show()
    sys.exit(app.exec_())