export default {
  extends: ['@commitlint/config-conventional'],
  rules: {
    'scope-enum': [
      1,
      'always',
      [
        'web',
        'backend',
        'ai-service',
        'capture',
        'contracts',
        'infra',
        'docs',
        'ci',
        'repo',
        'deps',
      ],
    ],
  },
};
