import { useReducer, useCallback } from "react";

export type ToastType = "success" | "error" | "info";

export interface Toast {
  id: string;
  message: string;
  type: ToastType;
}

type Action =
  | { type: "ADD"; toast: Toast }
  | { type: "REMOVE"; id: string };

function reducer(state: Toast[], action: Action): Toast[] {
  switch (action.type) {
    case "ADD":
      return [...state, action.toast];
    case "REMOVE":
      return state.filter((t) => t.id !== action.id);
  }
}

export function useToast() {
  const [toasts, dispatch] = useReducer(reducer, []);

  const removeToast = useCallback((id: string) => {
    dispatch({ type: "REMOVE", id });
  }, []);

  const addToast = useCallback(
    (message: string, type: ToastType = "info") => {
      const id = `${Date.now()}-${Math.random()}`;
      dispatch({ type: "ADD", toast: { id, message, type } });
      setTimeout(() => dispatch({ type: "REMOVE", id }), 4000);
    },
    []
  );

  return { toasts, addToast, removeToast };
}
