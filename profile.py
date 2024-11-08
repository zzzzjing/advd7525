# Import CloudLab libraries
import geni.portal as portal
import geni.rspec.pg as pg

# Initialize the Request object
request = portal.context.makeRequestRSpec()

# Node configuration
node = request.RawPC("node")
node.hardware_type = "d7525"  # Set node type
node.disk_image = "urn:publicid:IDN+emulab.net+image+emulab-ops:UBUNTU22-64-STD"  # Ubuntu 22.04

# GPU configuration
node.exclusive = True
node.addService(pg.Execute(shell="sh", command="sudo /usr/local/etc/emulab/mkextrafs.sh /mnt"))

# System update and Python environment setup
node.addService(pg.Execute(shell="sh", command="sudo apt update"))
node.addService(pg.Execute(shell="sh", command="sudo apt install -y python3.8 python3.8-venv python3-pip"))

# Install NVIDIA driver
node.addService(pg.Execute(shell="sh", command="sudo apt install -y nvidia-driver-510"))

# Install CUDA 11.7
node.addService(pg.Execute(shell="sh", command="sudo apt install -y cuda-11-7"))
node.addService(pg.Execute(shell="sh", command="sudo ln -s /usr/local/cuda-11.7 /usr/local/cuda"))

# Set CUDA toolkit paths
node.addService(pg.Execute(shell="sh", command="echo 'export PATH=/usr/local/cuda-11.7/bin:$PATH' >> ~/.bashrc"))
node.addService(pg.Execute(shell="sh", command="echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc"))
node.addService(pg.Execute(shell="sh", command="source ~/.bashrc"))

# Install PyTorch and deep learning dependencies
node.addService(pg.Execute(shell="sh", command="pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117"))
node.addService(pg.Execute(shell="sh", command="pip3 install numpy scipy pandas matplotlib"))

# Create Python virtual environment and install dependencies
node.addService(pg.Execute(shell="sh", command="python3.8 -m venv dl_env"))
node.addService(pg.Execute(shell="sh", command="source dl_env/bin/activate && pip install torch torchvision torchaudio numpy scipy pandas matplotlib"))

# Verify CUDA and PyTorch configuration
node.addService(pg.Execute(shell="sh", command="python3 -c 'import torch; print(\"CUDA available:\", torch.cuda.is_available()); print(\"CUDA device count:\", torch.cuda.device_count()); print(\"CUDA device name:\", torch.cuda.get_device_name(0))'"))

# Print completion message
node.addService(pg.Execute(shell="sh", command="echo 'CloudLab setup complete with Python 3.8, CUDA 11.7, and PyTorch'"))

# Output the configuration
portal.context.printRequestRSpec()
