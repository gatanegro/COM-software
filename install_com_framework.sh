#!/bin/bash

# COM Framework Software Installation Script
# This script installs the required dependencies and sets up the COM Framework software

echo "Installing COM Framework Software..."
echo "===================================="

# Create directories if they don't exist
mkdir -p ~/com_framework/test_results
mkdir -p ~/com_framework/visualizations

# Install required dependencies
echo "Installing dependencies..."
pip3 install numpy scipy matplotlib pandas scikit-learn PyQt5 seaborn mpmath

# Copy files to installation directory
echo "Copying files..."
cp com_framework_core_fixed2.py ~/com_framework/com_framework_core.py
cp com_visualization_fixed.py ~/com_framework/com_visualization.py
cp com_analysis_fixed.py ~/com_framework/com_analysis.py
cp com_gui.py ~/com_framework/com_gui.py
cp com_framework_app.py ~/com_framework/com_framework_app.py
cp com_tests.py ~/com_framework/com_tests.py

# Create launcher script
echo "Creating launcher..."
cat > ~/com_framework/run_com_framework.sh << 'EOF'
#!/bin/bash
cd ~/com_framework
python3 com_framework_app.py
EOF

# Make launcher executable
chmod +x ~/com_framework/run_com_framework.sh

# Create desktop shortcut
echo "Creating desktop shortcut..."
cat > ~/Desktop/COM_Framework.desktop << 'EOF'
[Desktop Entry]
Version=1.0
Type=Application
Name=COM Framework
Comment=Continuous Oscillatory Model Framework Explorer
Exec=~/com_framework/run_com_framework.sh
Icon=python
Terminal=false
Categories=Science;Math;
EOF

chmod +x ~/Desktop/COM_Framework.desktop

echo "Installation complete!"
echo "To run the COM Framework software, use one of the following methods:"
echo "1. Double-click the COM Framework icon on your desktop"
echo "2. Run ~/com_framework/run_com_framework.sh"
echo "3. Navigate to ~/com_framework and run python3 com_framework_app.py"
