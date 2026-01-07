# Running Tests from Local Machine

This guide explains how to test the SD-triton service from your local machine.

## Prerequisites

1. **Python 3** with `requests` library installed:
   ```bash
   pip install requests
   ```

2. **Network Access** to the server:
   - Server must be accessible from your local machine
   - Port 8700 must be open (check firewall rules)

## Usage

### Basic Usage

```bash
# Replace YOUR_SERVER_IP with your actual server IP address
python3 test_client_local.py your_audio.wav --server-url http://YOUR_SERVER_IP:8700
```

### Examples

#### 1. Connect to Remote Server
```bash
# If your server IP is 192.168.1.100
python3 test_client_local.py audio.wav --server-url http://192.168.1.100:8700

# If your server has a domain name
python3 test_client_local.py audio.wav --server-url http://your-server.com:8700
```

#### 2. Auto-detect Speakers
```bash
python3 test_client_local.py audio.wav --server-url http://YOUR_SERVER_IP:8700
```

#### 3. Specify Number of Speakers
```bash
python3 test_client_local.py audio.wav \
  --server-url http://YOUR_SERVER_IP:8700 \
  --num-speakers 2
```

#### 4. Pretty Print JSON Output
```bash
python3 test_client_local.py audio.wav \
  --server-url http://YOUR_SERVER_IP:8700 \
  --pretty
```

#### 5. Use Localhost (if running on same machine)
```bash
python3 test_client_local.py audio.wav --server-url http://localhost:8700
```

## Finding Your Server IP

### On Linux/Mac:
```bash
# Find server IP address
hostname -I
# or
ip addr show
```

### On Windows:
```bash
ipconfig
```

### From Local Machine:
```bash
# Test connectivity
curl http://YOUR_SERVER_IP:8700/v2/health/ready

# Or use ping
ping YOUR_SERVER_IP
```

## Firewall Configuration

Make sure port 8700 is open on your server:

### Ubuntu/Debian (UFW):
```bash
sudo ufw allow 8700/tcp
sudo ufw status
```

### CentOS/RHEL (firewalld):
```bash
sudo firewall-cmd --add-port=8700/tcp --permanent
sudo firewall-cmd --reload
```

### AWS EC2:
- Go to Security Groups
- Add inbound rule: Port 8700, Source: Your IP or 0.0.0.0/0

## Troubleshooting

### Connection Refused
```
❌ Connection Error: Could not connect to server
```

**Solutions:**
1. Verify server is running: `docker ps | grep sd-triton-server`
2. Check server IP is correct
3. Ensure firewall allows port 8700
4. Test with: `curl http://YOUR_SERVER_IP:8700/v2/health/ready`

### Timeout
```
❌ Timeout Error: Request took too long
```

**Solutions:**
1. Audio file might be too large
2. Server might be processing other requests
3. Network might be slow
4. Try with a smaller audio file first

### Server Not Ready
```
❌ Server is not ready or not accessible
```

**Solutions:**
1. Wait a bit longer (model loading takes 1-2 minutes)
2. Check server logs: `docker logs sd-triton-server`
3. Verify model loaded successfully
4. Restart server if needed

## Example Output

```
================================================================================
SD-triton Test Client (Local Machine)
================================================================================

Server URL: http://192.168.1.100:8700
Audio file: audio.wav

Checking server health...
✅ Server is ready!

Processing audio file: audio.wav
Auto-detecting number of speakers...

Reading and encoding audio file: audio.wav
Sending request to http://192.168.1.100:8700/v2/models/speaker_diarization/infer...

================================================================================
Speaker Diarization Results
================================================================================
Number of speakers detected: 2
Total segments: 5
Speakers: SPEAKER_00, SPEAKER_01

Segments:
--------------------------------------------------------------------------------
Segment 1:
  Time: 0.50s - 2.10s (duration: 1.60s)
  Speaker: SPEAKER_00

Segment 2:
  Time: 2.50s - 4.00s (duration: 1.50s)
  Speaker: SPEAKER_01
...
================================================================================
```

## Security Note

- The audio file is base64 encoded and sent over HTTP
- For production, consider using HTTPS
- Ensure your server is behind a firewall
- Don't expose the server to the public internet without proper security measures












