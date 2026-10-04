public static class SeedUtil
{
    // stable (does not depend on string.GetHashCode): FNV-1a over the salt and the four bytes of the seed
    public static int Derive(int seed, string salt)
    {
        unchecked
        {
            uint h = 2166136261u;
            foreach (char ch in salt) { h ^= ch; h *= 16777619u; }
            for (int i = 0; i < 4; i++) { h ^= (uint)((seed >> (8 * i)) & 0xFF); h *= 16777619u; }
            return (int)(h & 0x7FFFFFFFu);
        }
    }
}