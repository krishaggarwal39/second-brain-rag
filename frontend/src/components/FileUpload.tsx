import React, { useState, useRef, useEffect, useCallback } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { UploadCloud, AlertCircle, CheckCircle2, XCircle, Loader2, Link } from 'lucide-react';
import api from '../services/api';
import { apiErrorMessage, apiErrorStatus } from '../utils/errors';

const MAX_SIZE_MB = 10;
const MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024;
const JOB_POLL_INTERVAL_MS = 2000;
const JOB_POLL_TIMEOUT_MS = 300_000; // 5 minutes max wait

type IndexingState = 'idle' | 'uploading' | 'indexing' | 'done' | 'failed';

export default function FileUpload() {
  const [dragActive, setDragActive] = useState(false);
  const [activeTab, setActiveTab] = useState<'file' | 'url'>('file');
  const [urlInput, setUrlInput] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [indexingState, setIndexingState] = useState<IndexingState>('idle');
  const fileInputRef = useRef<HTMLInputElement>(null);
  const queryClient = useQueryClient();

  // Polls /documents/jobs/:jobId until completion or timeout.
  const pollJobStatus = useCallback(async (jobId: string): Promise<void> => {
    const deadline = Date.now() + JOB_POLL_TIMEOUT_MS;
    while (Date.now() < deadline) {
      await new Promise((r) => setTimeout(r, JOB_POLL_INTERVAL_MS));
      try {
        const res = await api.get(`/documents/jobs/${jobId}`);
        const status: string = res.data.status ?? 'unknown';
        if (status === 'completed') {
          setIndexingState('done');
          sessionStorage.removeItem('activeJobId');
          queryClient.invalidateQueries({ queryKey: ['documents'] });
          return;
        }
        if (status.startsWith('failed')) {
          const reason = status.replace('failed: ', '');
          setError(`Indexing failed: ${reason}`);
          setIndexingState('failed');
          sessionStorage.removeItem('activeJobId');
          return;
        }
        // still "processing" — keep polling
      } catch (e: unknown) {
        setError(apiErrorMessage(e, 'Failed to check job status'));
        setIndexingState('failed');
        sessionStorage.removeItem('activeJobId');
        return;
      }
    }
    // Timed out — treat as failed
    setError('Indexing timed out. The file may be too large or the server is busy.');
    setIndexingState('failed');
    sessionStorage.removeItem('activeJobId');
  }, [queryClient]);

  // Resume polling a job that was still running when the page was reloaded.
  useEffect(() => {
    const activeJobId = sessionStorage.getItem('activeJobId');
    if (activeJobId) {
      setIndexingState('indexing');
      setProgress(100);
      pollJobStatus(activeJobId);
    }
  }, [pollJobStatus]);

  const uploadMutation = useMutation({
    mutationFn: async (file: globalThis.File) => {
      setIndexingState('uploading');
      const formData = new FormData();
      formData.append('file', file);
      const res = await api.post('/documents/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (progressEvent) => {
          if (progressEvent.total) {
            const pct = Math.round((progressEvent.loaded * 100) / progressEvent.total);
            setProgress(pct);
          }
        },
      });
      return res.data as { job_id: string; status: string };
    },
    onSuccess: async (data) => {
      setIndexingState('indexing');
      setProgress(100);
      sessionStorage.setItem('activeJobId', data.job_id);
      await pollJobStatus(data.job_id);
    },
    onError: (err: unknown) => {
      setError(apiErrorMessage(err, 'Upload failed'));
      setIndexingState('failed');
      setProgress(0);
    },
  });

  const urlMutation = useMutation({
    mutationFn: async (url: string) => {
      setIndexingState('uploading');
      const res = await api.post('/documents/url', { url });
      return res.data as { job_id: string; status: string };
    },
    onSuccess: async (data) => {
      setIndexingState('indexing');
      setProgress(100);
      sessionStorage.setItem('activeJobId', data.job_id);
      await pollJobStatus(data.job_id);
    },
    onError: (err: unknown) => {
      if (apiErrorStatus(err) === 429) {
        setError('Rate limit reached (2 requests per minute). Please wait.');
      } else {
        setError(apiErrorMessage(err, 'URL ingestion failed'));
      }
      setIndexingState('failed');
      setProgress(0);
    },
  });

  const handleUrlSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    let finalUrl = urlInput.trim();
    if (!finalUrl) return;
    
    if (!finalUrl.startsWith('http://') && !finalUrl.startsWith('https://')) {
      finalUrl = 'https://' + finalUrl;
    }
    
    urlMutation.mutate(finalUrl);
  };

  const validateFile = (file: globalThis.File): boolean => {
    setError(null);
    setIndexingState('idle');

    if (file.size > MAX_SIZE_BYTES) {
      setError(`File size exceeds ${MAX_SIZE_MB}MB limit.`);
      return false;
    }

    const validTypes = [
      'application/pdf',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      'application/msword',
    ];
    if (!validTypes.includes(file.type) && !file.name.endsWith('.pdf') && !file.name.endsWith('.docx')) {
      setError('Only PDF and DOCX files are allowed.');
      return false;
    }

    return true;
  };

  const handleFiles = (files: FileList | null) => {
    if (!files || files.length === 0) return;
    const file = files[0];
    if (validateFile(file)) uploadMutation.mutate(file);
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') setDragActive(true);
    else if (e.type === 'dragleave') setDragActive(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files?.length) handleFiles(e.dataTransfer.files);
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files?.length) handleFiles(e.target.files);
  };

  const busy = indexingState === 'uploading' || indexingState === 'indexing';

  return (
    <div style={{ width: '100%' }}>
      <div style={{ display: 'flex', gap: '1rem', marginBottom: '1.5rem', borderBottom: '1px solid var(--border)' }}>
        <button
          onClick={() => { setActiveTab('file'); setError(null); }}
          style={{
            padding: '0.5rem 1rem',
            background: 'transparent',
            border: 'none',
            borderBottom: activeTab === 'file' ? '2px solid var(--accent-primary)' : '2px solid transparent',
            color: activeTab === 'file' ? 'var(--text-primary)' : 'var(--text-secondary)',
            fontWeight: activeTab === 'file' ? 600 : 400,
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            transition: 'all 0.2s'
          }}
          disabled={busy}
        >
          <UploadCloud size={18} /> File Upload
        </button>
        <button
          onClick={() => { setActiveTab('url'); setError(null); }}
          style={{
            padding: '0.5rem 1rem',
            background: 'transparent',
            border: 'none',
            borderBottom: activeTab === 'url' ? '2px solid var(--accent-primary)' : '2px solid transparent',
            color: activeTab === 'url' ? 'var(--text-primary)' : 'var(--text-secondary)',
            fontWeight: activeTab === 'url' ? 600 : 400,
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            transition: 'all 0.2s'
          }}
          disabled={busy}
        >
          <Link size={18} /> Web URL
        </button>
      </div>

      {activeTab === 'file' ? (
      <div
        className={`dropzone ${dragActive ? 'active' : ''}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => !busy && fileInputRef.current?.click()}
        style={{ cursor: busy ? 'not-allowed' : 'pointer' }}
      >
        <input
          ref={fileInputRef}
          type="file"
          style={{ display: 'none' }}
          accept=".pdf,.docx,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
          onChange={handleChange}
          disabled={busy}
        />

        <UploadCloud className="dropzone-icon" size={48} />
        <p style={{ fontSize: '1.1rem', fontWeight: 500, color: 'var(--text-primary)', marginBottom: '0.5rem' }}>
          Drag and drop your file here
        </p>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
          Supports PDF and DOCX up to {MAX_SIZE_MB}MB
        </p>

        {/* Upload progress bar */}
        {indexingState === 'uploading' && (
          <div style={{ width: '100%', maxWidth: '300px', margin: '1.5rem auto 0' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>
              <span>Uploading…</span>
              <span>{progress}%</span>
            </div>
            <div style={{ width: '100%', backgroundColor: 'rgba(255,255,255,0.1)', borderRadius: '99px', height: '8px' }}>
              <div style={{ backgroundColor: 'var(--accent-primary)', height: '8px', borderRadius: '99px', width: `${progress}%`, transition: 'width 0.3s ease' }} />
            </div>
          </div>
        )}

        {/* Indexing spinner */}
        {indexingState === 'indexing' && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '1.5rem', color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
            <Loader2 size={16} style={{ animation: 'spin 1s linear infinite' }} />
            <span>Indexing document… this may take a moment</span>
          </div>
        )}
      </div>
      ) : (
        <form onSubmit={handleUrlSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1rem', padding: '1rem', background: 'var(--bg-surface)', borderRadius: 'var(--radius-xl)', border: 'var(--glass-border)' }}>
          <div>
            <label style={{ display: 'block', marginBottom: '0.5rem', color: 'var(--text-primary)', fontWeight: 500 }}>
              Website URL
            </label>
            <input
              type="text"
              placeholder="https://example.com/article"
              value={urlInput}
              onChange={(e) => setUrlInput(e.target.value)}
              disabled={busy}
              style={{
                width: '100%',
                padding: '0.75rem 1rem',
                borderRadius: 'var(--radius-md)',
                border: '1px solid var(--border)',
                background: 'rgba(255, 255, 255, 0.05)',
                color: 'var(--text-primary)',
                outline: 'none',
              }}
            />
          </div>
          <button
            type="submit"
            disabled={busy || !urlInput.trim()}
            style={{
              padding: '0.75rem 1.5rem',
              borderRadius: 'var(--radius-md)',
              border: 'none',
              background: 'var(--accent-primary)',
              color: '#000',
              fontWeight: 600,
              cursor: busy || !urlInput.trim() ? 'not-allowed' : 'pointer',
              opacity: busy || !urlInput.trim() ? 0.7 : 1,
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
              gap: '0.5rem',
              transition: 'all 0.2s'
            }}
          >
            {indexingState === 'uploading' || indexingState === 'indexing' ? (
              <><Loader2 size={18} style={{ animation: 'spin 1s linear infinite' }} /> Scrape in progress...</>
            ) : (
              <><Link size={18} /> Scrape Website</>
            )}
          </button>
        </form>
      )}

      {error && (
        <div className="error-box" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <AlertCircle size={16} />
          <span>{error}</span>
        </div>
      )}

      {indexingState === 'done' && (
        <div style={{ background: 'rgba(0, 245, 212, 0.1)', border: '1px solid rgba(0, 245, 212, 0.3)', color: 'var(--accent-secondary)', padding: '1rem', borderRadius: 'var(--radius-md)', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <CheckCircle2 size={16} />
          <span>Indexed successfully — document is ready to query.</span>
        </div>
      )}

      {indexingState === 'failed' && !error && (
        <div style={{ background: 'rgba(255,80,80,0.1)', border: '1px solid rgba(255,80,80,0.3)', color: '#ff5050', padding: '1rem', borderRadius: 'var(--radius-md)', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <XCircle size={16} />
          <span>Indexing failed. Check server logs for details.</span>
        </div>
      )}
    </div>
  );
}
