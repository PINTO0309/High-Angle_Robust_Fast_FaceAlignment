// Enforce pnpm as the package manager (npm/yarn installs bypass the
// minimumReleaseAge quarantine configured in pnpm-workspace.yaml).
const ua = process.env.npm_config_user_agent ?? '';
if (!ua.startsWith('pnpm/')) {
  console.error('This project must be installed with pnpm (npm/yarn are blocked): run `pnpm install`.');
  process.exit(1);
}
