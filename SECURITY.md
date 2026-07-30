# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability within UniversalAI, please report it responsibly. We value the security of our users and appreciate your efforts to keep the project safe.

### How to Report

**Please do not report security vulnerabilities through public GitHub issues or pull requests.**

Instead, please use one of the following methods:

1. **GitHub Security Advisories**: [Report privately](https://github.com/6t9xstar/universal-ai/security/advisories/new)
2. **Email**: hello@universal-ai.dev
3. **Discord**: [Join our server](https://discord.gg/universal-ai) (send a DM to a maintainer)

### What to Include

When reporting a vulnerability, please include:

- A clear description of the vulnerability
- Steps to reproduce the issue
- The impact of the vulnerability
- Any mitigations you've discovered
- Your contact information (if you'd like to be credited)

### What to Expect

- We will acknowledge your report within 48 hours
- We will work with you to understand and validate the issue
- We will develop a fix and coordinate a release timeline
- We will credit you in our changelog (unless you wish to remain anonymous)

## Supported Versions

Security updates are applied to the latest release version. We encourage all users to keep their UniversalAI installation up to date.

## Security Best Practices

- Always use Environment Variables for API keys
- Never commit API keys or secrets to version control
- Regularly update your UniversalAI installation
- Review dependencies for known vulnerabilities with `pip audit`
- Use the ` CircuitBreakerMiddleware` to protect against cascading failures
- Enable `CostTrackingMiddleware` to monitor for unexpected usage spikes

## Dependency Security

We use `pip-audit` and GitHub Dependabot to monitor dependencies for known vulnerabilities.

## Disclosure Policy

We follow a coordinated disclosure policy. Once a fix has been developed and tested, we will release a security update and publicly disclose the vulnerability.