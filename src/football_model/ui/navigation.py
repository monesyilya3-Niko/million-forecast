"""Unified navigation state management.

Provides page history, filter persistence, and proper back navigation.
"""

from __future__ import annotations

import logging
from typing import Any

import streamlit as st

logger = logging.getLogger(__name__)

# Session state keys
NAV_PAGE = "nav_page"
PAGE_HISTORY = "page_history"
RETURN_PAGE = "return_page"
FILTERS = "page_filters"


def _init_state() -> None:
    if PAGE_HISTORY not in st.session_state:
        st.session_state[PAGE_HISTORY] = []
    if FILTERS not in st.session_state:
        st.session_state[FILTERS] = {}


def navigate_to(page_name: str, context: dict[str, Any] | None = None, *, set_return: bool = True) -> None:
    """Navigate to a page, setting return page and preserving context."""
    _init_state()
    current = st.session_state.get(NAV_PAGE, "⚽ 今日竞彩")

    if set_return and current != page_name:
        st.session_state[RETURN_PAGE] = current

    if current != page_name:
        history = st.session_state[PAGE_HISTORY]
        if not history or history[-1] != current:
            history.append(current)
            st.session_state[PAGE_HISTORY] = history[-10:]

    st.session_state[NAV_PAGE] = page_name

    if context:
        for key, value in context.items():
            st.session_state[key] = value

    st.rerun()


def go_back(default_page: str = "⚽ 今日竞彩") -> None:
    """Navigate back to the previous page."""
    _init_state()
    return_page = st.session_state.pop(RETURN_PAGE, default_page)
    st.session_state[NAV_PAGE] = return_page
    st.rerun()


def get_current_page() -> str:
    return st.session_state.get(NAV_PAGE, "⚽ 今日竞彩")


def get_return_page(default: str = "⚽ 今日竞彩") -> str:
    return st.session_state.get(RETURN_PAGE, default)


def remember_filter(page_key: str, filter_key: str, value: Any) -> None:
    _init_state()
    if page_key not in st.session_state[FILTERS]:
        st.session_state[FILTERS][page_key] = {}
    st.session_state[FILTERS][page_key][filter_key] = value


def restore_filter(page_key: str, filter_key: str, default: Any = None) -> Any:
    _init_state()
    return st.session_state.get(FILTERS, {}).get(page_key, {}).get(filter_key, default)


def render_back_button(label: str = "← 返回上一页", key: str = "back") -> bool:
    """Render a back button. Returns True if clicked."""
    return_page = get_return_page()
    if st.button(label, key=key):
        go_back(return_page)
        return True
    return False
