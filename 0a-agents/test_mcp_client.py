#!/usr/bin/env python3
import subprocess
import json
import sys

def send_json_rpc(process, message):
    """Send a JSON-RPC message to the MCP server and read the response"""
    json_message = json.dumps(message) + '\n'
    print(f"Sending: {json_message.strip()}")
    
    process.stdin.write(json_message.encode())
    process.stdin.flush()
    
    # Read response
    response_line = process.stdout.readline().decode().strip()
    print(f"Received: {response_line}")
    
    if response_line:
        try:
            return json.loads(response_line)
        except json.JSONDecodeError:
            print(f"Failed to parse JSON: {response_line}")
            return None
    return None

def main():
    # Start the MCP server process
    process = subprocess.Popen(
        ['python', 'weather_server.py'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd='/home/ubuntuser/llm-zoomcamp-learn/0a-agents'
    )
    
    try:
        # 1. Initialize
        init_message = {
            "jsonrpc": "2.0", 
            "id": 1, 
            "method": "initialize", 
            "params": {
                "protocolVersion": "2024-11-05", 
                "capabilities": {
                    "roots": {"listChanged": True}, 
                    "sampling": {}
                }, 
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            }
        }
        
        init_response = send_json_rpc(process, init_message)
        print("Initialize response:", init_response)
        print()
        
        # 2. Send initialized notification
        initialized_message = {
            "jsonrpc": "2.0", 
            "method": "notifications/initialized"
        }
        
        send_json_rpc(process, initialized_message)
        print("Initialized notification sent")
        print()
        
        # 3. List tools
        list_tools_message = {
            "jsonrpc": "2.0", 
            "id": 2, 
            "method": "tools/list"
        }
        
        tools_response = send_json_rpc(process, list_tools_message)
        print("Tools list response:", json.dumps(tools_response, indent=2))
        print()
        
        # 4. Call get_weather tool for Berlin
        call_tool_message = {
            "jsonrpc": "2.0", 
            "id": 3, 
            "method": "tools/call", 
            "params": {
                "name": "get_weather_tool", 
                "arguments": {"city": "Berlin"}
            }
        }
        
        weather_response = send_json_rpc(process, call_tool_message)
        print("Weather response:", json.dumps(weather_response, indent=2))
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        process.terminate()
        process.wait()

if __name__ == "__main__":
    main() 