
// A simple event emitter for cross-component communication.
// Used here to propagate Firestore permission errors to a global listener.

type EventMap = Record<string, any>;
type EventKey<T extends EventMap> = string & keyof T;
type EventReceiver<T> = (params: T) => void;

interface Emitter<T extends EventMap> {
  on<K extends EventKey<T>>(eventName: K, fn: EventReceiver<T[K]>): void;
  off<K extends EventKey<T>>(eventName: K, fn: EventReceiver<T[K]>): void;
  emit<K extends EventKey<T>>(eventName: K, params: T[K]): void;
}

export function createEmitter<T extends EventMap>(): Emitter<T> {
  const listeners: {
    [K in keyof T]?: Array<EventReceiver<T[K]>>;
  } = {};

  return {
    on(eventName, fn) {
      listeners[eventName] = (listeners[eventName] || []).concat(fn);
    },
    off(eventName, fn) {
      listeners[eventName] = (listeners[eventName] || []).filter(
        (f) => f !== fn
      );
    },
    emit(eventName, params) {
      (listeners[eventName] || []).forEach(function (fn) {
        fn(params);
      });
    },
  };
}

// App-specific error event type
import type { FirestorePermissionError } from './errors';

type ErrorEvents = {
  'permission-error': FirestorePermissionError;
};

// Global error emitter instance
export const errorEmitter = createEmitter<ErrorEvents>();
