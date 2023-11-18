using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using Ookii.Dialogs.Wpf;
using ViewModel;
using System.IO;

namespace MainUserInterface
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window, IUIServices
    {
        public MainWindow()
        {
            InitializeComponent();
            DataContext = new MainViewModel(this);
        }

        public List<string> ExtractFiles(string folderName, string format = "")
        {
            var extractedFiles = new List<string>();
            try
            {
                foreach (var file in Directory.EnumerateFiles(folderName))
                {
                    if (file.EndsWith(format))
                    {
                        extractedFiles.Add(file);
                    }
                }
            }
            catch (Exception e)
            {
                ReportError(e.Message);
            }
            return extractedFiles;
        }

        public string? GetFolderName()
        {
            VistaFolderBrowserDialog dialog = new VistaFolderBrowserDialog();
            if (dialog.ShowDialog() == true)
            {
                return dialog.SelectedPath;
            }
            return null;
        }

        public void ReportError(string message)
        {
            MessageBox.Show(message);
        }
    }
}
