// Keepsake web configuration. The CDK stack generates config.js with these fields from its
// outputs at deploy time (infra/constructs/web.py). Copy to config.js only for local testing.
// Nothing here is secret: the Cognito app client is a public PKCE client with no secret.
window.KEEPSAKE_CONFIG = Object.freeze({
  apiUrl: "https://abc123.execute-api.us-east-1.amazonaws.com/prod",
  cognitoDomain: "https://keepsake-123456789012.auth.us-east-1.amazoncognito.com",
  clientId: "exampleclientid",
  redirectUri: "https://d1234example.cloudfront.net/index.html",
});
