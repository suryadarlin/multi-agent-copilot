const API_BASE = 'https://multi-agent-copilot.onrender.com';

// ----------SECTION REQUEST HELPERS (avoid silent CORS/network failures)----------
// Network error helper (fetch throws TypeError on CORS/network)
async function apiRequest(method, path, body) {
  const url = API_BASE + path;
  try {
    return await fetch(url, {
      method,
      headers: { 'Content-Type': 'application/json' },
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch (err) {
    throw new Error(`Fetch failed (${method} ${url}). ${err?.message || String(err)}`);
  }
}




// ---------- SECTION 1 — AI Copilot ----------
const promptEl = document.getElementById('prompt');


const generateBtn = document.getElementById('generateBtn');
const btnText = document.getElementById('btnText');
const spinnerEl = document.getElementById('spinner');
const messageEl = document.getElementById('message');
const filesEl = document.getElementById('files');
const criticEl = document.getElementById('critic');
const elapsedEl = document.getElementById('elapsed');
const requestIdEl = document.getElementById('requestId');
const successEl = document.getElementById('success');

function setMessage(text, kind) {
  messageEl.textContent = text || '';
  messageEl.classList.remove('ok', 'err');
  if (kind) messageEl.classList.add(kind);
}

function setLoading(isLoading) {
  generateBtn.disabled = isLoading;
  spinnerEl.hidden = !isLoading;
  btnText.textContent = isLoading ? 'Generating' : 'Generate';
}

function renderFiles(generatedFiles) {
  // backend: generated_files is expected to be Dict[str, str]
  // but keep compatible with array/string
  filesEl.innerHTML = '';
  if (!generatedFiles) return;

  if (Array.isArray(generatedFiles)) {
    generatedFiles.forEach((f) => {
      const li = document.createElement('li');
      li.textContent = String(f);
      filesEl.appendChild(li);
    });
    return;
  }

  if (typeof generatedFiles === 'object') {
    Object.keys(generatedFiles).forEach((name) => {
      const li = document.createElement('li');
      li.textContent = name;
      filesEl.appendChild(li);
    });
  }
}

async function onGenerate() {
  const prompt = (promptEl.value || '').trim();
  if (!prompt) {
    setMessage('Enter a requirement.', 'err');
    return;
  }

  setMessage('', null);
  requestIdEl.textContent = '—';
  elapsedEl.textContent = '—';
  successEl.textContent = '—';
  criticEl.textContent = '{}';
  renderFiles([]);

  setLoading(true);

  try {
    const resp = await fetch(API_BASE + '/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt }),
    });

    const data = await resp.json().catch(() => ({}));

    if (!resp.ok) {
      const detail = data?.detail || data?.message || `Request failed (${resp.status})`;
      throw new Error(detail);
    }

    requestIdEl.textContent = String(data.request_id ?? '');
    successEl.textContent = String(data.success ?? false);
    renderFiles(data.generated_files);
    criticEl.textContent = JSON.stringify(data.critic_feedback || {}, null, 2);
    elapsedEl.textContent = typeof data.elapsed_ms === 'number' ? `${data.elapsed_ms} ms` : String(data.elapsed_ms ?? '—');

    setMessage(data.success ? 'Success.' : 'Failed.', data.success ? 'ok' : 'err');
  } catch (e) {
    renderFiles([]);
    criticEl.textContent = '{}';
    elapsedEl.textContent = '—';
    successEl.textContent = '—';
    requestIdEl.textContent = '—';
    setMessage(e?.message || String(e), 'err');
  } finally {
    setLoading(false);
  }
}

generateBtn.addEventListener('click', onGenerate);

// Optional: Enter key to trigger generation
promptEl.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') onGenerate();
});

// ---------- SECTION 2 — Student Management ----------
const studentForm = document.getElementById('studentForm');
const studentMsg = document.getElementById('studentMsg');
const studentNameEl = document.getElementById('studentName');
const studentRollNoEl = document.getElementById('studentRollNo');
const studentDeptEl = document.getElementById('studentDepartment');
const addStudentBtn = document.getElementById('addStudentBtn');
const refreshStudentsBtn = document.getElementById('refreshStudentsBtn');
const studentsTbody = document.getElementById('studentsTbody');
const studentsStatusEl = document.getElementById('studentsStatus');

function setStudentMsg(text, kind) {
  studentMsg.textContent = text || '';
  studentMsg.classList.remove('ok', 'err');
  if (kind) studentMsg.classList.add(kind);
}

function normalizeStudents(data) {
  if (Array.isArray(data)) return data;
  if (data && typeof data === 'object') {
    if (Array.isArray(data.items)) return data.items;
    if (Array.isArray(data.students)) return data.students;
    if (Array.isArray(data.data)) return data.data;
  }
  return [];
}

function getStudentId(s) {
  return s.id ?? s.student_id ?? s.roll_no ?? '';
}

function escapeHtml(str) {
  return String(str).replace(/[&<>"']/g, (m) => ({ '&': '&amp;', '<': '<', '>': '>', '"': '"', "'": '&#039;' }[m]));
}


async function loadStudents() {
  studentsTbody.innerHTML = '';
  studentsStatusEl.textContent = 'Loading...';
  setStudentMsg('', null);

  try {
    const res = await apiRequest('GET', '/students');
    if (!res.ok) throw new Error(`Failed to fetch students (HTTP ${res.status})`);
    const data = await res.json();
    console.log(data);
    const students = normalizeStudents(data);
    console.log("Students:", students);

    if (!students.length) {
      studentsStatusEl.textContent = 'No students found.';
      return;
    }

    studentsStatusEl.textContent = `Total: ${students.length}`;

    students.forEach((s) => {
      const id = getStudentId(s);
      const name = s.name ?? '';
      const rollNo = s.roll_no ?? s.rollNo ?? '';
      const dept = s.department ?? '';
      console.log("Adding row:", s);

      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${escapeHtml(String(id))}</td>
        <td>${escapeHtml(String(name))}</td>
        <td>${escapeHtml(String(rollNo))}</td>
        <td>${escapeHtml(String(dept))}</td>
        <td>
          <button class="miniBtn" type="button" data-id="${escapeHtml(String(id))}">Delete</button>
        </td>
      `;

      tr.querySelector('button[data-id]').addEventListener('click', async (e) => {
        const sid = e.currentTarget.getAttribute('data-id');
        if (!confirm('Delete this student?')) return;
        try {
          const delRes = await apiRequest('DELETE', `/students/${encodeURIComponent(sid)}`);
          if (!delRes.ok) throw new Error(`Delete failed (HTTP ${delRes.status})`);
          setStudentMsg('Deleted successfully.', 'ok');
          await loadStudents();
        } catch (err) {
          setStudentMsg(String(err.message || err), 'err');
        }
      });

      studentsTbody.appendChild(tr);
    });
  } catch (err) {
    studentsStatusEl.textContent = 'Failed to load students.';
    setStudentMsg(String(err.message || err), 'err');
  }
}

studentForm.addEventListener('submit', async (e) => {
  e.preventDefault();

  const name = (studentNameEl.value || '').trim();
  const roll_no = (studentRollNoEl.value || '').trim();
  const department = (studentDeptEl.value || '').trim();

  setStudentMsg('', null);

  try {
    // Requirement: POST /students with fields name, roll_no, department
    const payload = { name, roll_no, department };
    addStudentBtn.disabled = true;
    const res = await apiRequest('POST', '/students', payload);
    if (!res.ok) throw new Error(`Create failed (HTTP ${res.status})`);

    setStudentMsg('Student added successfully.', 'ok');
    studentForm.reset();
    await loadStudents();
  } catch (err) {
    setStudentMsg(String(err.message || err), 'err');
  } finally {
    addStudentBtn.disabled = false;
  }
});

refreshStudentsBtn.addEventListener('click', loadStudents);

// Initial fetch
document.addEventListener('DOMContentLoaded', () => {
  loadStudents();
});


