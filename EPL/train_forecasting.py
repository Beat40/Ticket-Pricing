from forecasting_engine_epl import EPLForecastingEngine
import asyncio
import json

async def train():
    engine = EPLForecastingEngine()
    res = await engine.train()
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    asyncio.run(train())
