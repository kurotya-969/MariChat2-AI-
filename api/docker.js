const { spawn } = require('child_process');

module.exports = async (req, res) => {
  try {
    // Dockerコンテナを起動
    const dockerProcess = spawn('docker', [
      'run', 
      '--rm', 
      '-p', '8501:8501',
      'marichat-app'
    ]);

    dockerProcess.stdout.on('data', (data) => {
      console.log(`Docker stdout: ${data}`);
    });

    dockerProcess.stderr.on('data', (data) => {
      console.error(`Docker stderr: ${data}`);
    });

    res.status(200).json({ 
      message: 'Docker container started successfully',
      status: 'running'
    });
  } catch (error) {
    console.error('Error starting Docker container:', error);
    res.status(500).json({ 
      error: 'Failed to start Docker container',
      details: error.message 
    });
  }
};