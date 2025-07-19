# weather_server.py
from fastmcp import FastMCP
from homework_agent import get_weather, set_weather

mcp = FastMCP("Weather Server 🌤️")

@mcp.tool()
def get_weather_tool(city: str) -> float:
    """
    Retrieves the temperature for a specified city.

    Parameters:
        city (str): The name of the city for which to retrieve weather data.

    Returns:
        float: The temperature associated with the city.
    """
    return get_weather(city)

@mcp.tool()
def set_weather_tool(city: str, temp: float) -> str:
    """
    Sets the temperature for a specified city.

    Parameters:
        city (str): The name of the city for which to set the weather data.
        temp (float): The temperature to associate with the city.

    Returns:
        str: A confirmation string 'OK' indicating successful update.
    """
    return set_weather(city, temp)

if __name__ == "__main__":
    mcp.run() 