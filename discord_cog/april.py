from redbot.core import commands
import aiohttp

class AprilCog(commands.Cog):
    """A cog that chats with an external API"""

    def __init__(self, bot):
        self.bot = bot
        self.api_url = "http://192.168.1.129/:8080/chat"

    @commands.command()
    async def chat(self, ctx, *, message: str):
        """Send a message to the external chat API and get a response."""
        async with aiohttp.ClientSession() as session:
            payload = {
                "user_message": message,
                "user_id": str(ctx.author.id)
            }
            async with session.post(self.api_url, json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    await ctx.send(data["response"])
                else:
                    await ctx.send(f"Error from chat API: {resp.status}")