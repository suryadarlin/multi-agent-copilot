// app.js - vanilla JS for Student Management frontend

// Change this if your FastAPI is hosted elsewhere.
window.FRONTEND_API_BASE = 'http://127.0.0.1:8000';

function apiUrl(path) {
  const base = window.FRONTEND_API_BASE || '';
  const cleanBase = base.replace(/\/$/, '');
  const cleanPath = path.startsWith('/') ? path : '/' + path;
  return cleanBase + cleanPath;
}

async function apiRequest(method, path, body) {
  const res = await fetch(apiUrl(path), {
    method,
    headers: {
      'Content-Type': 'application/json',
    },
    body: body ? JSON.stringify(body) : undefined,
  });
  return res;
}

// Public helpers used by pages
window.apiGet = (path) => apiRequest('GET', path);
window.apiDelete = (path) => apiRequest('DELETE', path);
window.apiCreateStudent = (payload) => apiRequest('POST', '/students', payload);

window.apiUpdateStudent = (id, payload) => apiRequest('PUT', `/students/${id}`, payload);

window.apiGetStudents = async () => {
  const res = await apiRequest('GET', '/students');
  if (!res.ok) throw new Error(`Failed to fetch students (HTTP ${res.status})`);
  return res.json();
};

