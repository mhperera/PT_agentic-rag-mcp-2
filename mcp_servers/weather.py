import httpx
from fastmcp import FastMCP
from langsmith import traceable
from core.env_loader import WEATHER_API_KEY

mcp = FastMCP("Weather Server")
BASE_URL = "https://api.weatherapi.com/v1/current.json"


@mcp.tool(name="get_weather", description="Get the weather location")
@traceable(name="Tool: Get Weather")
async def get_weather(location: str) -> str:
    params = {"key": WEATHER_API_KEY, "q": location, "aqi": "no"}
    headers = {"User-Agent": "MyWeatherApp/1.0 (contact@example.com)"}
    # return f"The weather in {location} is mostly sunny with light breeze"
    async with httpx.AsyncClient() as client:
        response = await client.get(BASE_URL, params=params, headers=headers)
        if response.status_code != 200:
            return f"Failed to get weather for {location}: {response.text}"

        data = response.json()
        current = data["current"]
        condition = current["condition"]["text"]
        temp = current["temp_c"]
        feels_like = current["feelslike_c"]
        humidity = current["humidity"]
        wind_kph = current["wind_kph"]

        return (
            f"Weather in {location}: {condition}, temperature {temp}°C "
            f"(feels like {feels_like}°C), humidity {humidity}%, wind speed {wind_kph} kph."
        )


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8000)
