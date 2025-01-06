from .april_cog import AprilCog


def setup(bot):
    """Load the AprilCog cog."""
    bot.add_cog(AprilCog(bot))
