"""
COM Framework User Interface

This module implements a graphical user interface for the Continuous Oscillatory
Model (COM) framework, providing access to LZ calculations, visualizations,
and cryptographic applications.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QTabWidget, 
                           QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, 
                           QLineEdit, QPushButton, QComboBox, QSpinBox, 
                           QDoubleSpinBox, QTextEdit, QFileDialog, QGroupBox,
                           QRadioButton, QCheckBox, QSlider, QProgressBar,
                           QSplitter, QFrame, QMessageBox, QSizePolicy, QScrollArea)
from PyQt5.QtGui import QFont, QIcon

# Import the core modules
from com_framework_core import LZModule, OctaveModule
from com_visualization import VisualizationModule

class MatplotlibCanvas(FigureCanvas):
    """Canvas for embedding Matplotlib figures in PyQt."""
    
    def __init__(self, figure=None, parent=None):
        if figure is None:
            figure = plt.figure(figsize=(5, 4), dpi=100)
        
        super().__init__(figure)
        self.setParent(parent)
        self.figure = figure
        self.setSizePolicy(QWidget.Expanding, QWidget.Expanding)
        self.updateGeometry()


class CalculationThread(QThread):
    """Thread for running calculations without freezing the UI."""
    
    finished = pyqtSignal(object)
    progress = pyqtSignal(int)
    
    def __init__(self, function, args=None, kwargs=None):
        super().__init__()
        self.function = function
        self.args = args or []
        self.kwargs = kwargs or {}
        
    def run(self):
        result = self.function(*self.args, **self.kwargs)
        self.finished.emit(result)


class COMFrameworkGUI(QMainWindow):
    """Main window for the COM Framework application."""
    
    def __init__(self):
        super().__init__()
        
        # Initialize modules
        self.lz_module = LZModule()
        self.octave_module = OctaveModule(self.lz_module)
        self.viz_module = VisualizationModule(self.lz_module, self.octave_module)
        
        # Set up the UI
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle('COM Framework')
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Add tabs
        self.add_dashboard_tab()
        self.add_lz_explorer_tab()
        self.add_octave_analysis_tab()
        self.add_crypto_tab()
        self.add_energy_patterns_tab()
        self.add_about_tab()
        
        # Status bar
        self.statusBar().showMessage('Ready')
        
        # Show the window
        self.show()
        
    def add_dashboard_tab(self):
        """Add the dashboard tab with overview visualizations."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Header
        header = QLabel("COM Framework Dashboard")
        header.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(header)
        
        # Description
        description = QLabel(
            "This dashboard provides an overview of the Continuous Oscillatory Model (COM) framework, "
            "showing key visualizations of LZ recursion, octave patterns, and applications."
        )
        description.setWordWrap(True)
        layout.addWidget(description)
        
        # Create a scroll area for the dashboard
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        
        # Container widget for the scroll area
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        
        # Dashboard canvas
        self.dashboard_canvas = MatplotlibCanvas()
        toolbar = NavigationToolbar(self.dashboard_canvas, self)
        
        scroll_layout.addWidget(toolbar)
        scroll_layout.addWidget(self.dashboard_canvas)
        
        # Controls
        controls_group = QGroupBox("Dashboard Controls")
        controls_layout = QHBoxLayout()
        
        # Initial value control
        initial_value_layout = QVBoxLayout()
        initial_value_label = QLabel("Initial Value:")
        self.initial_value_spin = QDoubleSpinBox()
        self.initial_value_spin.setRange(0.1, 10.0)
        self.initial_value_spin.setValue(1.0)
        self.initial_value_spin.setSingleStep(0.1)
        initial_value_layout.addWidget(initial_value_label)
        initial_value_layout.addWidget(self.initial_value_spin)
        controls_layout.addLayout(initial_value_layout)
        
        # Iterations control
        iterations_layout = QVBoxLayout()
        iterations_label = QLabel("Iterations:")
        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(5, 100)
        self.iterations_spin.setValue(20)
        iterations_layout.addWidget(iterations_label)
        iterations_layout.addWidget(self.iterations_spin)
        controls_layout.addLayout(iterations_layout)
        
        # Refresh button
        self.refresh_dashboard_btn = QPushButton("Refresh Dashboard")
        self.refresh_dashboard_btn.clicked.connect(self.refresh_dashboard)
        controls_layout.addWidget(self.refresh_dashboard_btn)
        
        # Save button
        self.save_dashboard_btn = QPushButton("Save Dashboard")
        self.save_dashboard_btn.clicked.connect(self.save_dashboard)
        controls_layout.addWidget(self.save_dashboard_btn)
        
        controls_group.setLayout(controls_layout)
        scroll_layout.addWidget(controls_group)
        
        # Set the scroll content
        scroll.setWidget(scroll_content)
        
        # Add tab to tab widget
        self.tabs.addTab(tab, "Dashboard")
        
        # Initial dashboard update
        self.refresh_dashboard()
        
    def refresh_dashboard(self):
        """Update the dashboard visualization."""
        self.statusBar().showMessage('Generating dashboard...')
        
        # Get parameters
        initial_value = self.initial_value_spin.value()
        iterations = self.iterations_spin.value()
        
        # Create calculation thread
        self.calc_thread = CalculationThread(
            self.viz_module.create_dashboard,
            args=[initial_value, iterations]
        )
        
        # Connect signals
        self.calc_thread.finished.connect(self.update_dashboard_canvas)
        
        # Start calculation
        self.calc_thread.start()
        
    def update_dashboard_canvas(self, figure):
        """Update the dashboard canvas with the new figure."""
        # Clear the old figure
        self.dashboard_canvas.figure.clear()
        
        # Copy the axes from the new figure to the canvas figure
        for ax in figure.get_axes():
            self.dashboard_canvas.figure.add_axes(ax)
        
        # Update the canvas
        self.dashboard_canvas.draw()
        
        # Close the new figure to free memory
        plt.close(figure)
        
        self.statusBar().showMessage('Dashboard updated')
        
    def save_dashboard(self):
        """Save the dashboard visualization to a file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Dashboard", "", "PNG Files (*.png);;All Files (*)"
        )
        
        if file_path:
            self.dashboard_canvas.figure.savefig(file_path, dpi=150, bbox_inches='tight')
            self.statusBar().showMessage(f'Dashboard saved to {file_path}')
        
    def add_lz_explorer_tab(self):
        """Add the LZ explorer tab for detailed LZ analysis."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Header
        header = QLabel("LZ Constant Explorer")
        header.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(header)
        
        # Description
        description = QLabel(
            "Explore the LZ constant (1.23498), its derivation through recursive functions, "
            "stability properties, and convergence behavior."
        )
        description.setWordWrap(True)
        layout.addWidget(description)
        
        # Split view: controls on left, visualization on right
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        # Left panel - Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # LZ Information group
        lz_info_group = QGroupBox("LZ Information")
        lz_info_layout = QGridLayout()
        
        lz_info_layout.addWidget(QLabel("LZ Constant:"), 0, 0)
        self.lz_value_label = QLabel(f"{self.lz_module.LZ:.8f}")
        self.lz_value_label.setFont(QFont("Courier", 10, QFont.Bold))
        lz_info_layout.addWidget(self.lz_value_label, 0, 1)
        
        lz_info_layout.addWidget(QLabel("HQS Threshold:"), 1, 0)
        self.hqs_value_label = QLabel(f"{self.lz_module.HQS:.8f}")
        self.hqs_value_label.setFont(QFont("Courier", 10))
        lz_info_layout.addWidget(self.hqs_value_label, 1, 1)
        
        lz_info_layout.addWidget(QLabel("Stability at LZ:"), 2, 0)
        stability = self.lz_module.stability_at_point(self.lz_module.LZ)
        self.stability_label = QLabel(f"{stability:.8f}")
        self.stability_label.setFont(QFont("Courier", 10))
        lz_info_layout.addWidget(self.stability_label, 2, 1)
        
        lz_info_group.setLayout(lz_info_layout)
        left_layout.addWidget(lz_info_group)
        
        # Derivation group
        derivation_group = QGroupBox("LZ Derivation")
        derivation_layout = QVBoxLayout()
        
        # Initial value
        init_layout = QHBoxLayout()
        init_layout.addWidget(QLabel("Initial Value:"))
        self.derive_initial_spin = QDoubleSpinBox()
        self.derive_initial_spin.setRange(0.1, 10.0)
        self.derive_initial_spin.setValue(1.0)
        self.derive_initial_spin.setSingleStep(0.1)
        init_layout.addWidget(self.derive_initial_spin)
        derivation_layout.addLayout(init_layout)
        
        # Max iterations
        iter_layout = QHBoxLayout()
        iter_layout.addWidget(QLabel("Max Iterations:"))
        self.derive_iterations_spin = QSpinBox()
        self.derive_iterations_spin.setRange(10, 1000)
        self.derive_iterations_spin.setValue(100)
        iter_layout.addWidget(self.derive_iterations_spin)
        derivation_layout.addLayout(iter_layout)
        
        # Precision
        precision_layout = QHBoxLayout()
        precision_layout.addWidget(QLabel("Precision:"))
        self.derive_precision_combo = QComboBox()
        for i in range(1, 11):
            self.derive_precision_combo.addItem(f"1e-{i}", 10**(-i))
        self.derive_precision_combo.setCurrentIndex(9)  # 1e-10
        precision_layout.addWidget(self.derive_precision_combo)
        derivation_layout.addLayout(precision_layout)
        
        # Derive button
        self.derive_button = QPushButton("Derive LZ")
        self.derive_button.clicked.connect(self.derive_lz)
        derivation_layout.addWidget(self.derive_button)
        
        # Results
        derivation_layout.addWidget(QLabel("Derivation Results:"))
        self.derive_results = QTextEdit()
        self.derive_results.setReadOnly(True)
        self.derive_results.setMaximumHeight(100)
        derivation_layout.addWidget(self.derive_results)
        
        derivation_group.setLayout(derivation_layout)
        left_layout.addWidget(derivation_group)
        
        # Visualization controls
        viz_group = QGroupBox("Visualization")
        viz_layout = QVBoxLayout()
        
        # Visualization type
        viz_layout.addWidget(QLabel("Visualization Type:"))
        self.viz_type_combo = QComboBox()
        self.viz_type_combo.addItems([
            "Recursive Function", 
            "Convergence to LZ", 
            "Stability Analysis",
            "Fixed Points"
        ])
        self.viz_type_combo.currentIndexChanged.connect(self.update_lz_visualization)
        viz_layout.addWidget(self.viz_type_combo)
        
        # Update button
        self.update_viz_button = QPushButton("Update Visualization")
        self.update_viz_button.clicked.connect(self.update_lz_visualization)
        viz_layout.addWidget(self.update_viz_button)
        
        viz_group.setLayout(viz_layout)
        left_layout.addWidget(viz_group)
        
        # Add stretch to push everything up
        left_layout.addStretch()
        
        # Right panel - Visualization
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Visualization canvas
        self.lz_canvas = MatplotlibCanvas()
        self.lz_toolbar = NavigationToolbar(self.lz_canvas, self)
        
        right_layout.addWidget(self.lz_toolbar)
        right_layout.addWidget(self.lz_canvas)
        
        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 700])  # Initial sizes
        
        # Add tab to tab widget
        self.tabs.addTab(tab, "LZ Explorer")
        
        # Initial visualization
        self.update_lz_visualization()
        
    def derive_lz(self):
        """Derive the LZ constant using the recursive function."""
        initial_value = self.derive_initial_spin.value()
        max_iterations = self.derive_iterations_spin.value()
        precision = self.derive_precision_combo.currentData()
        
        self.statusBar().showMessage('Deriving LZ...')
        self.derive_button.setEnabled(False)
        
        # Create calculation thread
        self.calc_thread = CalculationThread(
            self.lz_module.derive_lz,
            args=[initial_value, max_iterations, precision]
        )
        
        # Connect signals
        self.calc_thread.finished.connect(self.show_derive_results)
        
        # Start calculation
        self.calc_thread.start()
        
    def show_derive_results(self, results):
        """Show the results of LZ derivation."""
        derived_lz, sequence, iterations = results
        
        # Format results
        result_text = f"Derived LZ: {derived_lz:.8f}\n"
        result_text += f"Iterations: {iterations}\n"
        result_text += f"Final Difference: {abs(derived_lz - sequence[-2]):.10f}\n"
        result_text += f"Error from known LZ: {abs(derived_lz - self.lz_module.LZ):.10f}"
        
        # Update results display
        self.derive_results.setText(result_text)
        
        # Update visualization if showing convergence
        if self.viz_type_combo.currentText() == "Convergence to LZ":
            self.update_lz_visualization()
            
        self.statusBar().showMessage('LZ derivation complete')
        self.derive_button.setEnabled(True)
        
    def update_lz_visualization(self):
        """Update the LZ visualization based on selected type."""
        viz_type = self.viz_type_combo.currentText()
        
        self.statusBar().showMessage(f'Generating {viz_type} visualization...')
        
        # Clear the current figure
        self.lz_canvas.figure.clear()
        ax = self.lz_canvas.figure.add_subplot(111)
        
        if viz_type == "Recursive Function":
            # Plot the recursive function
            x_values = np.linspace(0, 3, 1000)
            y_values = [self]



















































































































































































































