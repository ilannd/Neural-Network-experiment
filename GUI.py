import wx
import time

from Test import IrisTest, Perception, Hopfile, RBF1, RBF2, RBF3

ID_EXIT = 200
ID_ABOUT = 201
ID_MR = 100
Version = "0.1"
ReleaseDate = "2020-11-11"


class MainFrame(wx.Frame):
    def __init__(self, parent, id, title):
        wx.Frame.__init__(self, parent, id, title, size=(500, 300))
        # 状态栏的创建
        self.setupStatusBar()
        # 显示按钮功能
        self.initUi()
        # 菜单栏的创建
        self.setupMenuBar()

    def initUi(self):
        # 显示按钮功能
        self.button1 = wx.Button(self, -1, u"实验一", (100, 20), (60, 30))  # (20,20)表示在Frame里的左上角坐标位置，(60,30)表示Button大小
        self.Bind(wx.EVT_BUTTON, self.OnClick,
                  self.button1)  # 将wx.EVT_BUTTON绑定到self.buttonOK上，通过self.OnClick这个行为来实现它的功能

        self.button2 = wx.Button(self, -1, u"实验二", (100, 80), (60, 30))
        self.Bind(wx.EVT_BUTTON, self.OnClick, self.button2)

        self.button3 = wx.Button(self, -1, u"实验三(1)", (200, 20), (60, 30))
        self.Bind(wx.EVT_BUTTON, self.OnClick, self.button3)

        self.button4 = wx.Button(self, -1, u"实验三(2)", (200, 80), (60, 30))
        self.Bind(wx.EVT_BUTTON, self.OnClick, self.button4)

        self.button5 = wx.Button(self, -1, u"实验三(3)", (300, 20), (60, 30))
        self.Bind(wx.EVT_BUTTON, self.OnClick, self.button5)

        self.button6 = wx.Button(self, -1, u"实验四", (300, 80), (60, 30))
        self.Bind(wx.EVT_BUTTON, self.OnClick, self.button6)


    def setupStatusBar(self):
        # 状态栏的创建
        sb = self.CreateStatusBar(2)  # 创建状态栏，2的意思是有两个内容，分成两个
        self.SetStatusWidths([-1, -2])  # 分为两个，比例为1:2
        self.SetStatusText("Ready", 0)  # Ready是里面的内容，0表示第一个
        # timer
        self.timer = wx.PyTimer(self.Notify)  # derived from wx.Timer   #Notify函数设置显示的时间格式
        self.timer.Start(1000, wx.TIMER_CONTINUOUS)  # 1000表示1秒钟开始，后面的表示一直在跑不停
        self.Notify()

    def setupMenuBar(self):
        # 创建菜单栏
        # 主菜单
        menubar = wx.MenuBar()
        # 子菜单：退出（Quit)
        fmenu = wx.Menu()
        fmenu.Append(ID_EXIT, u'退出(&Q)', 'Terminate the program')
        # 将子菜单添加到文件(File)中
        menubar.Append(fmenu, u'文件(&F)')
        # 子菜单：关于（About)
        hmenu = wx.Menu()
        # 将子菜单添加到帮助(Help)中
        hmenu.Append(ID_ABOUT, u'关于(&A)', 'More information about this program')
        menubar.Append(hmenu, u'帮助(&H)')
        self.SetMenuBar(menubar)  # 加入frame里面
        # 菜单中子菜单，时间行为的绑定即实现
        wx.EVT_MENU(self, ID_EXIT, self.OnMenuExit)
        wx.EVT_MENU(self, ID_ABOUT, self.OnMenuAbout)
        wx.EVT_CLOSE(self, self.OnCloseWindow)

    def OnClick(self, event):
        if event.GetEventObject() == self.button1:
            IrisTest.iris()
        elif event.GetEventObject() == self.button2:
            Perception.main()
        elif event.GetEventObject() == self.button3:
            RBF1.main()
        elif event.GetEventObject() == self.button4:
            RBF2.main()
        elif event.GetEventObject() == self.button5:
            RBF3.main()
        elif event.GetEventObject() == self.button6:
            Hopfile.main()
        else:
            print("NO button is clicked")

    def Notify(self):
        t = time.localtime(time.time())
        st = time.strftime('%Y-%m-%d  %H:%M:%S', t)
        self.SetStatusText(st, 1)  # 将时间显示在后一个

    def OnMenuExit(self, event):
        self.Close()

    def OnMenuAbout(self, event):
        dlg = AboutDialog(None, -1)
        dlg.ShowModal()
        dlg.Destory()

    def OnCloseWindow(self, event):
        self.Destroy()


class AboutDialog(wx.Dialog):
    def __init__(self, parent, id):
        wx.Dialog.__init__(self, parent, id, 'About Me', size=(200, 200))
        # 布局管理
        self.sizer1 = wx.BoxSizer(wx.VERTICAL)
        self.sizer1.Add(wx.StaticText(self, -1, u"神经网络实验报告"), 0, wx.ALIGN_CENTER_HORIZONTAL | wx.TOP, border=20)
        self.sizer1.Add(wx.StaticText(self, -1, u"(C) 2020 吕玉玺"), 0, wx.ALIGN_CENTER_HORIZONTAL | wx.TOP, border=10)
        self.sizer1.Add(wx.StaticText(self, -1, u"Version %s , %s" % (Version, ReleaseDate)), 0,
                        wx.ALIGN_CENTER_HORIZONTAL | wx.TOP, border=10)
        self.sizer1.Add(wx.StaticText(self, -1, u"Author : 吕玉玺"), 0, wx.ALIGN_CENTER_HORIZONTAL | wx.TOP, border=10)
        self.sizer1.Add(wx.Button(self, wx.ID_OK), 0, wx.ALIGN_CENTER | wx.BOTTOM, border=20)
        self.SetSizer(self.sizer1)


class App(wx.App):
    def __init__(self):
        # 如果要重写__init__,必须要调用wx.App的__init__,否则OnInit方法不会被调用
        super(self.__class__, self).__init__()
        # wx.App.__init__(self)

    def OnInit(self):
        self.version = u"报告"  # 这里把"第二课“用Unicode编码，表示Unicode字符串,一般用在中文字符串前面，防止因为源码储存格式问题，导致再次使用时出现乱码
        self.title = u"神经网络实验" + self.version
        frame = MainFrame(None, -1, self.title)
        frame.Show(True)

        return True


if __name__ == "__main__":
    app = App()
    app.MainLoop()
