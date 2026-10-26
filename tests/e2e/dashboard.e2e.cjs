// Browser end-to-end check of the dashboard (API on :8000, Streamlit on :8501 over the real data).
// Run: npx playwright install chromium   (once)
//      SHOTS_DIR=e2e-screenshots node tests/e2e/dashboard.e2e.cjs
// Visits every tab, approves the top suggestion through the form, asks the copilot, and fails on any page error.
const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width: 1400, height: 1000 } });
  const errors = [];
  p.on('pageerror', e => errors.push(String(e)));
  await p.goto(process.env.DASHBOARD_URL || 'http://localhost:8501', { waitUntil: 'networkidle' });
  await p.getByText('Forecast origin (end of day)').waitFor({ timeout: 60000 });
  await p.waitForTimeout(3000);
  const shot = async n => p.screenshot({ path: `${process.env.SHOTS_DIR || 'e2e-screenshots'}/${n}.png`, fullPage: true });
  await shot('1_overview');
  const tab = async name => { await p.getByRole('tab', { name }).click(); await p.waitForTimeout(3500); };
  await tab('Forecasts'); await shot('2_forecasts');
  await tab('Unusual sales'); await shot('3_unusual');
  await tab('Stockout risk & approvals'); await shot('4_risk');
  await p.getByLabel('Your name (recorded in the audit log)').fill('Dashboard Tester');
  await p.getByLabel('Note (optional)').fill('e2e check');
  await p.getByRole('button', { name: 'Approve' }).click();
  await p.getByText(/approved by Dashboard Tester/).waitFor({ timeout: 30000 });
  await shot('5_approved');
  await tab('Copilot');
  await p.getByRole('textbox', { name: 'Question' }).fill('Which products are at risk of stockout in the next seven days?');
  await p.getByRole('button', { name: 'Ask' }).click();
  await p.getByText(/every number matches a tool result|unverified numbers/).waitFor({ timeout: 60000 });
  await shot('6_copilot');
  const exc = await p.locator('[data-testid="stException"]').count();
  console.log(JSON.stringify({ pageErrors: errors, streamlitExceptions: exc }));
  if (errors.length || exc) process.exit(1);
  await b.close();
})().catch(e => { console.error('E2E FAILED:', e.message); process.exit(1); });
