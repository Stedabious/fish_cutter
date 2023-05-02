from PyQt5 import QtWidgets

# from controller import Login_Window_controller
# from controller import Run_Window_Controller
from controller_new import Login_Window_controller
from controller_new import Run_Window_Controller

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    login_window = Run_Window_Controller()
    login_window.show()
    sys.exit(app.exec_())
    
