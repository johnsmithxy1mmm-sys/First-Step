/** @type {import('next').NextConfig} */
export default {
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
