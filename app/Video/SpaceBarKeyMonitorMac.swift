//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2025 Apple Inc. All Rights Reserved.
//
// Local monitor so Space works without the preview view being key (List/TextField
// would otherwise take focus and SwiftUI onKeyPress never fires).
//

#if os(macOS)

import AppKit
import SwiftUI

struct SpaceBarKeyMonitorMac: NSViewRepresentable {
    var isEnabled: Bool
    var onSpace: () -> Void

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    func makeNSView(context: Context) -> NSView {
        let v = NSView()
        v.frame = .zero
        context.coordinator.installIfNeeded()
        return v
    }

    func updateNSView(_ nsView: NSView, context: Context) {
        context.coordinator.isEnabled = isEnabled
        context.coordinator.onSpace = onSpace
    }

    final class Coordinator: NSObject {
        var monitor: Any?
        var isEnabled: Bool = true
        var onSpace: (() -> Void)?

        func installIfNeeded() {
            guard monitor == nil else { return }
            monitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
                guard let self, self.isEnabled, event.keyCode == 49 else { return event }
                let mods = event.modifierFlags.intersection([.command, .control, .option, .function])
                if !mods.isEmpty { return event }
                if self.textInputIsFirstResponder { return event }
                self.onSpace?()
                return nil
            }
        }

        /// Avoid stealing Space from prompt fields.
        var textInputIsFirstResponder: Bool {
            guard let r = NSApp.keyWindow?.firstResponder else { return false }
            if r is NSTextView { return true }
            if r is NSTextField { return true }
            return false
        }

        deinit {
            if let m = monitor {
                NSEvent.removeMonitor(m)
            }
        }
    }
}

#endif
