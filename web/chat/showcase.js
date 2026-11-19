"use strict";

// Showcase: one long shared context (an invented support runbook) asked about over several
// turns. The self-hosted engine keeps the runbook's KV cache, so only the first turn pays to
// process it; Bedrock on-demand charges for the whole runbook again on every turn.
window.SHOWCASE = {
  title: "Showcase: repeated long context",
  maxTokens: 60,
  questions: [
    "Using only the runbook, what is the first thing to check when a customer cannot log in?",
    "According to the runbook, when must a ticket be escalated to tier 2?",
    "Per the runbook, what should the agent say when closing a resolved ticket?",
  ],
  system: `You are a support assistant. Answer in at most two sentences, using only this runbook.

RUNBOOK: Example Co. customer support, version 4.2 (fictional, for demonstration only)

1. Scope
This runbook covers first-line support for the Example Co. web application: account access,
billing questions, data exports, notifications and integrations. Tier 1 agents handle every new
ticket. Tier 2 engineers handle anything that needs database access, code changes or a refund
above 200 dollars. Security issues always go to the security on-call engineer, never to tier 2.

2. Login problems
2.1 First check the status page. If there is an active incident affecting sign-in, link the
customer to the incident and set the ticket to "waiting on incident". Do not troubleshoot
individual accounts during an active sign-in incident.
2.2 If there is no incident, confirm the email address on the account. About a third of login
tickets are customers using a work address on a personal account or the other way round.
2.3 Ask whether the customer sees an error code. Code E101 means the password is wrong: send the
reset link. Code E204 means the account is locked after ten failed attempts: it unlocks itself
after thirty minutes, or tier 1 can unlock it from the admin console after verifying identity.
Code E309 means single sign-on failed: ask for the identity provider name and escalate to tier 2.
2.4 Identity verification needs two of: the billing postcode, the last four digits of the card
on file, or the date of the last invoice. Never accept a screenshot as verification.

3. Billing
3.1 Invoices are issued on the first day of each month and are available under Settings, Billing.
3.2 Tier 1 can issue refunds up to 200 dollars for duplicate charges or for outages longer than
four hours that are listed on the status page. Anything else needs tier 2 approval.
3.3 Plan downgrades take effect at the end of the billing period; upgrades take effect at once
and are charged pro rata.
3.4 Failed payments are retried on days 3, 7 and 14. After the third failure the workspace becomes
read-only, and all data is kept for 60 days.

4. Data exports
4.1 Workspace owners can export all data as CSV or JSON from Settings, Data. Exports above 5 GB
are delivered as a download link by email within 24 hours.
4.2 If an export is missing after 24 hours, check the email address on the workspace owner
account, then escalate to tier 2 with the workspace ID.

5. Notifications and integrations
5.1 Most missing-email reports are caused by spam filters. Ask the customer to allow messages from
the notifications domain before investigating further.
5.2 Integration errors show a code starting with I. Codes I1xx are configuration problems the
customer can fix by reconnecting the integration. Codes I5xx are service-side and must be
escalated to tier 2.

6. Escalation
6.1 Escalate to tier 2 when: the issue needs database access or a code change; the customer
reports data loss; an error code says to escalate (E309, I5xx); a refund above 200 dollars is
requested; or the ticket has been open for more than 48 hours without progress.
6.2 Every escalation must include the workspace ID, the steps already tried, the exact error code
or message, and the time the problem was first seen.
6.3 Security reports (suspected account takeover, exposed credentials, unexpected admin users)
go straight to the security on-call engineer by pager, with the ticket marked confidential.

7. Tone and closing
7.1 Use the customer's name, keep replies short, and give one clear next step per message.
7.2 When closing a resolved ticket, summarise what was fixed, say that replying to the email
reopens the ticket within 7 days, and thank the customer for their patience.
7.3 Never promise dates for new features. Say that the request has been passed to the product team.`,
};
