/** @type {import('next').NextConfig} */
export default {
  // Standalone bundles only the files the server actually reaches, which is
  // what makes the runtime image small enough to be worth shipping.
  output: 'standalone',
  async rewrites() {
    // The backend owns the §6 contract; the browser never talks to the
    // Python engine directly.
    return [
      {
        source: '/api/:path*',
        destination: `${process.env.BACKEND_URL ?? 'http://127.0.0.1:8080'}/api/:path*`,
      },
    ];
  },
};
