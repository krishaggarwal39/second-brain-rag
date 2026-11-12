interface ValidationDetail {
  msg?: string;
}

interface ErrorLike {
  message?: unknown;
  response?: { status?: unknown; data?: { detail?: unknown } };
}

function asErrorLike(err: unknown): ErrorLike | undefined {
  return err !== null && typeof err === 'object' ? (err as ErrorLike) : undefined;
}

/** HTTP status of a failed API call, if there was a response. */
export function apiErrorStatus(err: unknown): number | undefined {
  const status = asErrorLike(err)?.response?.status;
  return typeof status === 'number' ? status : undefined;
}

/**
 * Human-readable message for any thrown value (axios errors, Error, or plain objects).
 * Understands FastAPI `detail` as a string or a validation-error list.
 */
export function apiErrorMessage(err: unknown, fallback: string): string {
  const e = asErrorLike(err);
  const detail = e?.response?.data?.detail;
  if (typeof detail === 'string' && detail) return detail;
  if (Array.isArray(detail) && detail.length) {
    return detail
      .map((d: ValidationDetail | string) => (typeof d === 'string' ? d : d?.msg || JSON.stringify(d)))
      .join(', ');
  }
  // A known HTTP status maps to the caller's specific fallback, not a transport message.
  if (apiErrorStatus(err) !== undefined) return fallback;
  if (typeof e?.message === 'string' && e.message) return e.message;
  return fallback;
}
