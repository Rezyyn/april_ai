from redbot.core.bot import Red
from redbot.core import commands
import aiohttp

from .april_cog import AprilCog


async def setup(bot: Red) -> None:
    """Load Dice cog."""
    cog = AprilCog(bot)
    await bot.add_cog(cog)